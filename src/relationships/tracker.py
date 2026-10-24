"""Storage of relationship records and the glue to Mantella's conversation flow.

Files live next to the paragraph summaries:
    <conversations folder>/<world_id>/<NPC name> - <ref_id>/<NPC name>_relationship.json
(or the legacy `<NPC name>/` folder if that is the one the summaries already use).
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from typing import Any, Callable

from src import utils
from src.character_manager import Character
from src.characters_manager import Characters
from src.llm.message_thread import message_thread
from src.llm.messages import Message, SystemMessage, UserMessage
from src.relationships import record as rec
from src.relationships import render, updater
from src.remember.remembering import Remembering

logger = utils.get_logger()

MIN_TALK_MESSAGES = 2
MAX_BYSTANDERS = 4  # nearby NPCs who get a record per conversation (closest first, as sent by the game)


class RelationshipStore:
    """Reads and writes one JSON file per NPC and world."""

    def __init__(self, conversation_folder_path: str) -> None:
        self.__base = conversation_folder_path
        self.__locks: dict[str, threading.Lock] = {}
        self.__locks_guard = threading.Lock()

    def path_for(self, npc_name: str, npc_ref_id: str, world_id: str) -> str:
        base_name = utils.remove_trailing_number(npc_name)
        world_folder = os.path.join(self.__base, world_id)
        name_ref_folder = os.path.join(world_folder, f"{base_name} - {npc_ref_id}")
        legacy_folder = os.path.join(world_folder, base_name)
        folder = legacy_folder if (not os.path.exists(name_ref_folder) and os.path.exists(legacy_folder)) else name_ref_folder
        return os.path.join(folder, f"{base_name}_relationship.json").replace(os.sep, "/")

    def lock_for(self, path: str) -> threading.Lock:
        with self.__locks_guard:
            return self.__locks.setdefault(path, threading.Lock())

    def load(self, npc_name: str, npc_ref_id: str, world_id: str, player_name: str = "") -> dict[str, Any]:
        path = self.path_for(npc_name, npc_ref_id, world_id)
        raw: Any = None
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Could not read relationship file {path} ({e}); starting a fresh record. The old file is kept as .bak")
                try:
                    os.replace(path, path + ".bak")
                except OSError:
                    pass
        return rec.normalize_record(raw, utils.remove_trailing_number(npc_name), npc_ref_id, player_name)

    def find_ref_id(self, npc_name: str, world_id: str) -> str | None:
        """ref_id of the single existing '<name> - <ref_id>' folder for this NPC in this world (None if none or several).
        Returns "" if only a legacy '<name>' folder exists (path_for then resolves to it)."""
        base_name = utils.remove_trailing_number(npc_name)
        world_folder = os.path.join(self.__base, world_id)
        if not os.path.isdir(world_folder):
            return None
        prefix = f"{base_name} - "
        refs = [entry[len(prefix):] for entry in os.listdir(world_folder)
                if entry.startswith(prefix) and os.path.isdir(os.path.join(world_folder, entry))]
        if len(refs) == 1:
            return refs[0]
        if not refs and os.path.isdir(os.path.join(world_folder, base_name)):
            return ""
        return None

    def exists(self, npc_name: str, npc_ref_id: str, world_id: str) -> bool:
        return os.path.exists(self.path_for(npc_name, npc_ref_id, world_id))

    def save(self, record: dict[str, Any], npc_name: str, npc_ref_id: str, world_id: str) -> str:
        path = self.path_for(npc_name, npc_ref_id, world_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path


class RelationshipTracker:
    """Loads records for prompts and updates them at conversation end."""

    def __init__(self, store: RelationshipStore, client: Any, game_name: str, language: str,
                 prompt_template: str = updater.DEFAULT_UPDATE_PROMPT, token_budget: int = 150,
                 decay_per_day: float = 0.0, affects_trust: bool = True, run_in_background: bool = True,
                 bystander_records: bool = False, resolve_ref_id: Callable[[str], str | None] | None = None,
                 lookup_bio: Callable[[str], str] | None = None, track_lies: bool = False) -> None:
        self.store = store
        self.__client = client
        self.__game_name = game_name
        self.__language = language
        self.__prompt_template = prompt_template or updater.DEFAULT_UPDATE_PROMPT
        self.token_budget = token_budget
        self.decay_per_day = decay_per_day
        self.affects_trust = affects_trust
        self.__background = run_in_background
        # Nearby NPCs who only overheard the conversation also get a record (same LLM call)
        self.bystander_records = bystander_records
        self.__resolve_ref_id = resolve_ref_id
        self.__lookup_bio = lookup_bio
        # Trust and honesty (src/trust): caught lies + credibility in the update call
        self.track_lies = track_lies
        # Messages already processed per NPC during the current conversation (reloads re-send the tail)
        self.__seen: dict[tuple[str, str, str], Counter] = {}
        # Memory checkpoints (src/memory_checkpoint) mark saved messages themselves and switch this off
        self.dedupe_reloads = True
        self.__threads: list[threading.Thread] = []

    # ---------- prompt side ----------
    def _count_tokens(self, text: str) -> int:
        try:
            return int(self.__client.get_count_tokens(text))
        except Exception:
            return render.approx_tokens(text)

    def get_prompt_text(self, npcs: list[Character], world_id: str, player_name: str, current_day: float | None) -> str:
        records = []
        for npc in npcs:
            if npc.is_player_character or not self.store.exists(npc.name, npc.ref_id, world_id):
                continue
            records.append(self.store.load(npc.name, npc.ref_id, world_id, player_name))
        return render.render_records(records, player_name, current_day, self.token_budget, self.decay_per_day, self._count_tokens)

    def get_opinion(self, npc: Character, world_id: str, current_day: float | None) -> int | None:
        if not self.store.exists(npc.name, npc.ref_id, world_id):
            return None
        record = self.store.load(npc.name, npc.ref_id, world_id)
        if not rec.has_met(record):
            return None
        return rec.decayed_opinion(record, current_day, self.decay_per_day)

    def adjust_trust(self, base_trust: str, npc: Character, world_id: str, current_day: float | None) -> str:
        if not self.affects_trust:
            return base_trust
        try:
            return render.adjust_trust(base_trust, self.get_opinion(npc, world_id, current_day))
        except Exception as e:
            logger.error(f"Relationship trust adjustment failed: {e}")
            return base_trust

    # ---------- update side ----------
    def update_after_conversation(self, groups: list[tuple[list[Character], message_thread]], world_id: str,
                                  player_name: str, witnesses: list[str], game_day: float | None,
                                  is_final: bool = True, bystanders: list[str] | None = None) -> None:
        """Entry point used by the rememberer. Runs in a background thread unless configured otherwise."""
        if self.__background:
            thread = threading.Thread(target=self._safe_update, args=(groups, world_id, player_name, witnesses, game_day, is_final, bystanders),
                                      name="relationship-update", daemon=False)
            self.__threads = [t for t in self.__threads if t.is_alive()] + [thread]
            thread.start()
        else:
            self._safe_update(groups, world_id, player_name, witnesses, game_day, is_final, bystanders)

    def wait_for_pending_updates(self, timeout: float | None = None) -> None:
        for thread in list(self.__threads):
            thread.join(timeout)

    def _safe_update(self, *args: Any) -> None:
        try:
            self.update_groups(*args)
        except Exception as e:
            logger.error(f"Relationship update failed: {e}")

    def _new_messages(self, keys: list[tuple[str, str, str]], messages: list[Message]) -> list[Message]:
        """Drop messages that were already processed in an earlier (reload) update of this conversation.
        Reloads summarise the whole thread and then keep its tail, so the tail would otherwise be counted twice."""
        if not self.dedupe_reloads:
            return list(messages)
        remaining = Counter(self.__seen.get(keys[0], Counter()))
        fresh = []
        for m in messages:
            content = m.get_formatted_content()
            if remaining[content] > 0:
                remaining[content] -= 1
                continue
            fresh.append(m)
        for key in keys:
            seen = self.__seen.setdefault(key, Counter())
            for m in fresh:
                seen[m.get_formatted_content()] += 1
        return fresh

    def resolve_bystanders(self, names: list[str], world_id: str, exclude: set[str]) -> list[tuple[str, str, str]]:
        """(name, ref_id, bio) for nearby NPCs that can be matched to one specific NPC. Unresolvable names
        (eg several 'Whiterun Guard's and none talked to before) are skipped: a record keyed by name only
        would be shared by all of them."""
        result: list[tuple[str, str, str]] = []
        for name in dict.fromkeys(n for n in names if n and n not in exclude):
            if len(result) >= MAX_BYSTANDERS:
                break
            ref_id = self.store.find_ref_id(name, world_id)
            if ref_id is None and self.__resolve_ref_id is not None:
                try:
                    ref_id = self.__resolve_ref_id(name) or None
                except Exception as e:
                    logger.debug(f"Could not resolve bystander {name}: {e}")
                    ref_id = None
            if ref_id is None:
                logger.info(f"Relationship: bystander {name} skipped (cannot be matched to one NPC).")
                continue
            bio = ""
            if self.__lookup_bio is not None:
                try:
                    bio = self.__lookup_bio(name) or ""
                except Exception:
                    bio = ""
            result.append((name, str(ref_id), bio))
        return result

    def update_groups(self, groups: list[tuple[list[Character], message_thread]], world_id: str, player_name: str,
                      witnesses: list[str], game_day: float | None, is_final: bool = True,
                      bystanders: list[str] | None = None) -> None:
        pending: list[tuple[list[Character], list[Message]]] = []
        for npcs, thread in groups:
            npcs = [n for n in npcs if not n.is_player_character]
            if not npcs:
                continue
            keys = [(world_id, n.name, str(n.ref_id)) for n in npcs]
            talk = [m for m in thread.get_talk_only() if not isinstance(m, SystemMessage)]
            fresh = self._new_messages(keys, talk)
            if is_final:
                for key in keys:
                    self.__seen.pop(key, None)
            if len(fresh) < MIN_TALK_MESSAGES or not any(isinstance(m, UserMessage) for m in fresh):
                logger.info(f"Relationship update skipped for {', '.join(n.name for n in npcs)}: not enough new dialogue.")
                continue
            pending.append((npcs, fresh))
        resolved: list[tuple[str, str, str]] = []
        if self.bystander_records and bystanders and pending:
            participants = {n.name for npcs, _ in groups for n in npcs} | {player_name}
            resolved = self.resolve_bystanders(bystanders, world_id, participants)
        # Bystanders ride along with the group that heard the most (normally the only group): still one call
        longest = max(range(len(pending)), key=lambda i: len(pending[i][1])) if pending else -1
        for i, (npcs, fresh) in enumerate(pending):
            self.update_records(npcs, message_thread.transform_to_text(fresh), world_id, player_name, witnesses, game_day,
                                bystanders=resolved if i == longest else None)

    def update_records(self, npcs: list[Character], transcript: str, world_id: str, player_name: str,
                       witnesses: list[str], game_day: float | None,
                       bystanders: list[tuple[str, str, str]] | None = None) -> dict[str, dict[str, Any]]:
        """One LLM call for all `npcs` (who heard the same transcript) and the `bystanders` ((name, ref_id, bio)
        of nearby NPCs who overheard it); merges and saves each record."""
        bystanders = [b for b in (bystanders or []) if b[0] not in {n.name for n in npcs}]
        paths = sorted({self.store.path_for(n.name, n.ref_id, world_id) for n in npcs}
                       | {self.store.path_for(name, ref, world_id) for name, ref, _ in bystanders})
        locks = [self.store.lock_for(p) for p in paths]
        for lock in locks:
            lock.acquire()
        try:
            records = {n.name: self.store.load(n.name, n.ref_id, world_id, player_name) for n in npcs}
            bystander_records = {name: self.store.load(name, ref, world_id, player_name) for name, ref, _ in bystanders}
            npc_names = {n.name for n in npcs} | set(bystander_records)
            others = [w for w in witnesses if w not in npc_names and w != player_name]
            system_prompt, user_content = updater.build_messages(
                self.__prompt_template, self.__game_name, self.__language, player_name,
                [(n.name, n.bio, records[n.name]) for n in npcs], others, transcript, game_day,
                bystanders=[(name, bio, bystander_records[name]) for name, _, bio in bystanders], track_lies=self.track_lies)
            thread = message_thread(None, system_prompt)
            thread.add_message(UserMessage(None, user_content))
            logger.log(23, f"Relationship update prompt: {user_content}")
            parsed = updater.request_update(self.__client, thread, self.track_lies)
            if parsed is None:
                return {}
            updates = _match_updates(parsed.get("npcs", []), [n.name for n in npcs] + list(bystander_records))
            saved: dict[str, dict[str, Any]] = {}
            for npc in npcs:
                update = updates.get(npc.name, {})
                # Everyone else in the conversation is a witness of what happened to this NPC
                for grudge in update.get("new_grudges") or []:
                    if isinstance(grudge, dict):
                        present = [n.name for n in npcs if n.name != npc.name]
                        grudge["witnesses"] = list(dict.fromkeys((grudge.get("witnesses") or []) + present))
                new_record = rec.apply_update(records[npc.name], update, game_day, self.decay_per_day)
                if player_name:
                    new_record["player_name"] = player_name
                path = self.store.save(new_record, npc.name, npc.ref_id, world_id)
                logger.info(f"Relationship of {npc.name} towards {player_name or 'the player'}: {new_record['opinion']:+d} ({new_record['opinion_label']}). Saved to {path}")
                saved[npc.name] = new_record
            for name, ref, _ in bystanders:
                update = updates.get(name)
                if not update:
                    continue  # the model returned nothing for this bystander: leave the record alone
                new_record = rec.apply_update(bystander_records[name], update, game_day, self.decay_per_day, bystander=True)
                if player_name:
                    new_record["player_name"] = player_name
                path = self.store.save(new_record, name, ref, world_id)
                logger.info(f"Relationship of bystander {name} towards {player_name or 'the player'}: {new_record['opinion']:+d} ({new_record['opinion_label']}). Saved to {path}")
                saved[name] = new_record
            return saved
        finally:
            for lock in reversed(locks):
                lock.release()


def _match_updates(entries: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, Any]]:
    """Map LLM entries to NPC names (exact, case-insensitive, then trailing-number-insensitive).
    A single unnamed entry for a single NPC is accepted."""
    result: dict[str, dict[str, Any]] = {}
    lowered = {n.lower(): n for n in names}
    base = {utils.remove_trailing_number(n).lower(): n for n in names}
    for entry in entries:
        raw_name = str(entry.get("name", "")).strip()
        name = raw_name if raw_name in names else lowered.get(raw_name.lower()) or base.get(utils.remove_trailing_number(raw_name).lower())
        if name and name not in result:
            result[name] = entry
    if not result and len(names) == 1 and len(entries) == 1:
        result[names[0]] = entries[0]
    return result


class RelationshipRememberer(Remembering):
    """Wraps the normal rememberer (Summaries): summaries work exactly as upstream, and after they
    are saved the relationship records are updated. Unknown attributes are forwarded to the wrapped object."""

    def __init__(self, inner: Remembering, tracker: RelationshipTracker) -> None:
        self._inner = inner
        self.relationship_tracker = tracker

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def get_prompt_text(self, characters: list[Character], world_id: str) -> str:
        return self._inner.get_prompt_text(characters, world_id)

    def save_conversation_state(self, messages: message_thread, npcs_to_summarize: list[Character], npcs_in_conversation: Characters,
                                world_id: str, is_reload=False, pending_shares: list[tuple[str, str, str]] | None = None,
                                end_timestamp: float | None = None, is_radiant: bool = False):
        self._inner.save_conversation_state(messages, npcs_to_summarize, npcs_in_conversation, world_id, is_reload,
                                            pending_shares, end_timestamp, is_radiant)
        if is_radiant:
            return  # NPC-to-NPC conversations: the player is not there
        try:
            groups = build_groups(self._inner, messages, npcs_to_summarize, npcs_in_conversation)
            witnesses = _witnesses(npcs_in_conversation)
            player_name = npcs_in_conversation.get_player_name() or ""
            bystanders = _bystanders(npcs_in_conversation) if getattr(self.relationship_tracker, "bystander_records", False) else None
            self.relationship_tracker.update_after_conversation(groups, world_id, player_name, witnesses, end_timestamp,
                                                                is_final=not is_reload, bystanders=bystanders)
        except Exception as e:
            logger.error(f"Relationship update could not be started: {e}")


def build_groups(summaries: Any, messages: message_thread, npcs_to_update: list[Character],
                 npcs_in_conversation: Characters) -> list[tuple[list[Character], message_thread]]:
    """Group NPCs that heard exactly the same messages (one LLM call per group), reusing the
    per-NPC thread logic of Summaries when available."""
    names = {c.name for c in npcs_to_update if not c.is_player_character}
    if hasattr(summaries, "get_threads_for_summarization") and hasattr(summaries, "group_shared_threads"):
        threads = summaries.get_threads_for_summarization(messages, npcs_in_conversation)
        threads = {name: params for name, params in threads.items() if name in names}
        groups = []
        for group in summaries.group_shared_threads(threads):
            chars = [c for c in npcs_to_update if c.name in group]
            if chars:
                groups.append((chars, threads[group[0]].messages))
        if groups:
            return groups
    chars = [c for c in npcs_to_update if not c.is_player_character]
    return [(chars, messages)] if chars else []


def _witnesses(npcs_in_conversation: Characters) -> list[str]:
    names = []
    try:
        names.extend(c.name for c in npcs_in_conversation.get_all_characters_since_start() if not c.is_player_character)
    except Exception:
        pass
    try:
        names.extend(npcs_in_conversation.get_nearby_npc_names())
    except Exception:
        pass
    return list(dict.fromkeys(n for n in names if n))


def _bystanders(npcs_in_conversation: Characters) -> list[str]:
    """Nearby NPCs who were never part of the conversation."""
    try:
        members = {c.name for c in npcs_in_conversation.get_all_characters_since_start()}
    except Exception:
        members = set()
    try:
        nearby = npcs_in_conversation.get_nearby_npc_names()
    except Exception:
        return []
    return [n for n in dict.fromkeys(nearby) if n and n not in members]
