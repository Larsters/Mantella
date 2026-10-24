"""Crash-safe memory: summaries and relationship updates are written during a conversation, not only at its end.

`CheckpointRememberer` wraps the normal rememberer (Summaries, or W6's RelationshipRememberer around it).
- Every N player turns, and once after X idle minutes, it saves the messages since the last checkpoint:
  one summary call (appended to the same `<NPC>_summary_N.txt`) plus, with NPC Relationships on, the usual
  relationship update (merged into the same record, under W6's per-file locks). This runs in a background thread.
- Every saved message gets a mark (`mantella_checkpoint`: NPC name -> fingerprint of its content). Marks
  survive the deep copies a reload makes. Every later save (the next checkpoint, a reload, an NPC leaving,
  the end of the conversation) replaces messages already saved for an NPC by a placeholder, so nothing is
  summarised or counted twice. If a message changed after it was marked (an NPC reply that was still
  streaming), it is saved again in full rather than lost.
- A small journal (`data/<game>/memory_checkpoints/active_conversation.json`) holds the conversation's
  unsaved messages. If the server itself dies, the next start saves them (`recover_unfinished`).
When nothing was checkpointed, the end-of-conversation save is passed through unchanged (upstream behaviour).
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

from src import utils
from src.llm.message_thread import message_thread
from src.llm.messages import AssistantMessage, Message, SystemMessage, UserMessage
from src.remember.remembering import Remembering

logger = utils.get_logger()

MARK_ATTR = "mantella_checkpoint"
PLACEHOLDER_TEXT = "[already saved by a memory checkpoint]"
# Summaries skips normal (non-reload) saves with fewer than 5 messages; checkpoints use the same bar so a
# checkpoint never marks messages that the summary would then silently drop.
MIN_CHECKPOINT_MESSAGES = 5
JOURNAL_NAME = "active_conversation.json"


# ---------- message marks ----------
def _sentences(message: Message) -> list[Any] | None:
    return getattr(message, "_AssistantMessage__sentences", None)


def fingerprint(message: Message) -> str:
    """Content of a message independent of display flags (is_multi_npc_message)."""
    sentences = _sentences(message)
    if sentences is not None:
        return "A:" + "␞".join(str(getattr(s, "text", "")) for s in sentences)
    return f"{type(message).__name__[0]}:{message.text}"


def is_talk(message: Message) -> bool:
    """User/assistant lines that summaries and relationship updates read (not system-generated ones)."""
    return isinstance(message, (UserMessage, AssistantMessage)) and not message.is_system_generated_message


def is_saved_for(message: Message, npc_name: str) -> bool:
    marks = getattr(message, MARK_ATTR, None)
    return bool(marks) and marks.get(npc_name) == fingerprint(message)


def mark_saved(messages: list[Message], npc_names: list[str]) -> None:
    for message in messages:
        marks = dict(getattr(message, MARK_ATTR, None) or {})
        fp = fingerprint(message)
        for name in npc_names:
            marks[name] = fp
        setattr(message, MARK_ATTR, marks)


def unmark(messages: list[Message], npc_names: list[str]) -> None:
    for message in messages:
        marks = dict(getattr(message, MARK_ATTR, None) or {})
        for name in npc_names:
            marks.pop(name, None)
        setattr(message, MARK_ATTR, marks)


def has_any_mark(messages: Any) -> bool:
    return any(getattr(messages[i], MARK_ATTR, None) for i in range(len(messages)))


def _speaker(message: Message, player_name: str) -> str:
    if isinstance(message, UserMessage):
        return player_name or "Player"
    sentences = _sentences(message) or []
    for s in sentences:
        name = getattr(getattr(s, "speaker", None), "name", "")
        if name:
            return name
    return ""


class _Plan:
    """One save for a group of NPCs who have the same unsaved messages."""

    def __init__(self, npcs: list[Any], indices: tuple[int, ...], had_checkpoint: bool) -> None:
        self.npcs = npcs
        self.indices = indices
        self.had_checkpoint = had_checkpoint
        self.thread: message_thread | None = None
        self.messages: list[Message] = []

    @property
    def names(self) -> list[str]:
        return [n.name for n in self.npcs]


def plan_saves(messages: Any, cutoff: int, npcs: list[Any]) -> list[_Plan]:
    """Group `npcs` by the talk messages (before `cutoff`) not yet saved for them, and build for each group a
    thread of the same length in which saved messages are placeholders (so participation indices still fit)."""
    cutoff = min(cutoff, len(messages))
    talk = [i for i in range(cutoff) if is_talk(messages[i])]
    groups: dict[tuple[int, ...], _Plan] = {}
    for npc in npcs:
        if getattr(npc, "is_player_character", False):
            continue
        kept = tuple(i for i in talk if not is_saved_for(messages[i], npc.name))
        had = len(kept) < len(talk)
        plan = groups.get(kept)
        if plan is None:
            plan = groups[kept] = _Plan([], kept, had)
        plan.npcs.append(npc)
        plan.had_checkpoint = plan.had_checkpoint or had
    placeholder = SystemMessage(PLACEHOLDER_TEXT, None)
    for plan in groups.values():
        keep = set(plan.indices)
        thread = message_thread(None, None)
        for i in range(cutoff):
            message = messages[i]
            if is_talk(message) and i not in keep:
                thread.add_message(placeholder)
            else:
                thread.add_message(message)
        plan.thread = thread
        plan.messages = [messages[i] for i in plan.indices]
    return list(groups.values())


# ---------- snapshots ----------
class FrozenCharacters:
    """The parts of `Characters` that Summaries and the relationship update read, copied at one moment so a
    background save does not see later joins and leaves."""

    def __init__(self, participation_log: list[tuple[str, str, int]], all_characters: list[Any],
                 player_name: str | None, nearby: list[str]) -> None:
        self._log = list(participation_log)
        self._all = list(all_characters)
        self._player_name = player_name
        self._nearby = list(nearby)

    @classmethod
    def of(cls, characters: Any) -> "FrozenCharacters":
        def call(name: str, default: Any) -> Any:
            try:
                return getattr(characters, name)()
            except Exception:
                return default
        return cls(call("get_participation_log", []), call("get_all_characters_since_start", []),
                   call("get_player_name", None), call("get_nearby_npc_names", []))

    def get_participation_log(self) -> list[tuple[str, str, int]]:
        return list(self._log)

    def get_all_characters_since_start(self) -> list[Any]:
        return list(self._all)

    def get_non_player_characters(self) -> list[Any]:
        return [c for c in self._all if not getattr(c, "is_player_character", False)]

    def get_player_name(self) -> str | None:
        return self._player_name

    def get_nearby_npc_names(self) -> list[str]:
        return list(self._nearby)

    def get_pending_shares(self) -> list[tuple[str, str, str]]:
        return []

    def clear_pending_shares(self) -> None:
        pass


class RecoveredCharacter:
    def __init__(self, name: str, ref_id: str, bio: str = "", is_player_character: bool = False) -> None:
        self.name = name
        self.ref_id = ref_id
        self.bio = bio
        self.is_player_character = is_player_character


class RecoveredUserMessage(UserMessage):
    def __init__(self, content: str) -> None:
        super().__init__(None, content)
        self._content = content

    def get_formatted_content(self) -> str:
        return self._content

    def get_dict_formatted_string(self) -> str:
        return str({"role": "user", "content": self._content})


class RecoveredAssistantMessage(AssistantMessage):
    def __init__(self, content: str) -> None:
        super().__init__(None)
        self._content = content

    def get_formatted_content(self) -> str:
        return self._content

    def get_dict_formatted_string(self) -> str:
        return str({"role": "assistant", "content": self._content})


class _Conversation:
    def __init__(self, context: Any, messages: Any, conversation_id: str | None) -> None:
        self.context = context
        self.messages = messages
        self.conversation_id = conversation_id or time.strftime("%Y%m%d-%H%M%S")
        self.turns_since_checkpoint = 0
        self.checkpoints = 0
        self.ended = False
        self.timer: threading.Timer | None = None


def _conversation_id() -> str | None:
    try:
        import src.usage_logger as usage_logger
        return getattr(usage_logger, "_context", {}).get("conversation_id")
    except Exception:
        return None


class CheckpointRememberer(Remembering):
    """See module docstring. Unknown attributes are forwarded to the wrapped rememberer."""

    def __init__(self, inner: Remembering, journal_folder: str, every_n_turns: int = 6, idle_minutes: float = 3,
                 run_in_background: bool = True, timer_factory: Callable[..., Any] | None = None) -> None:
        self._inner = inner
        self.journal_folder = journal_folder
        self.every_n_turns = max(0, int(every_n_turns))
        self.idle_seconds = max(0.0, float(idle_minutes) * 60)
        self._background = run_in_background
        self._timer_factory = timer_factory or threading.Timer
        self._state_lock = threading.RLock()
        self._save_lock = threading.Lock()  # one save at a time: summary files are appended without locks
        self._conversation: _Conversation | None = None
        self._threads: list[threading.Thread] = []

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def get_prompt_text(self, characters: list[Any], world_id: str) -> str:
        return self._inner.get_prompt_text(characters, world_id)

    # ---------- triggers ----------
    def on_player_turn(self, messages: Any, context: Any) -> None:
        """Called by Conversation after a normal player line was added. Never blocks on an LLM call."""
        try:
            if not getattr(context.config, "conversation_summary_enabled", True):
                return
            with self._state_lock:
                conv = self._conversation
                if conv is None or conv.context is not context:
                    self._cancel_timer(conv)
                    conv = self._conversation = _Conversation(context, messages, _conversation_id())
                conv.messages = messages
                conv.turns_since_checkpoint += 1
                self._arm_idle_timer(conv)
                due = self.every_n_turns > 0 and conv.turns_since_checkpoint >= self.every_n_turns
            if due:
                # Everything before the player's new line: the replies before it are complete
                cutoff = max((i for i in range(len(messages)) if isinstance(messages[i], UserMessage)), default=len(messages))
                self.checkpoint(conv, cutoff, "every %d turns" % self.every_n_turns)
            self._write_journal(conv)
        except Exception as e:
            logger.error(f"Memory checkpoint: turn hook failed: {e}")

    def _arm_idle_timer(self, conv: _Conversation) -> None:
        self._cancel_timer(conv)
        if self.idle_seconds <= 0:
            return
        timer = self._timer_factory(self.idle_seconds, self._on_idle, args=(conv,))
        timer.daemon = True
        conv.timer = timer
        timer.start()

    @staticmethod
    def _cancel_timer(conv: _Conversation | None) -> None:
        if conv is not None and conv.timer is not None:
            conv.timer.cancel()
            conv.timer = None

    def _on_idle(self, conv: _Conversation) -> None:
        try:
            with self._state_lock:
                if conv is not self._conversation or conv.ended:
                    return
                conv.timer = None
            self.checkpoint(conv, len(conv.messages), "idle")
            self._write_journal(conv)
        except Exception as e:
            logger.error(f"Memory checkpoint: idle checkpoint failed: {e}")

    def checkpoint(self, conv: _Conversation, cutoff: int, reason: str) -> bool:
        """Mark the unsaved messages before `cutoff` and save them in the background. Returns True if started."""
        with self._state_lock:
            if conv.ended:
                return False
            context = conv.context
            characters = context.npcs_in_conversation
            if not characters.contains_player_character():
                return False  # radiant
            plans = []
            for plan in plan_saves(conv.messages, cutoff, characters.get_non_player_characters()):
                if len(plan.messages) < MIN_CHECKPOINT_MESSAGES or not any(isinstance(m, UserMessage) for m in plan.messages):
                    continue
                mark_saved(plan.messages, plan.names)
                plans.append(plan)
            if not plans:
                return False
            conv.turns_since_checkpoint = 0
            conv.checkpoints += 1
            frozen = FrozenCharacters.of(characters)
            world_id = context.world_id
            game_days = getattr(context, "game_days", None)
        logger.info(f"Memory checkpoint ({reason}): saving {sum(len(p.messages) for p in plans)} new messages for "
                    f"{', '.join(n for p in plans for n in p.names)} in the background.")
        if self._background:
            thread = threading.Thread(target=self._run_checkpoint, args=(plans, frozen, world_id, game_days, conv),
                                      name="memory-checkpoint", daemon=False)
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
            thread.start()
        else:
            self._run_checkpoint(plans, frozen, world_id, game_days, conv)
        return True

    def _run_checkpoint(self, plans: list[_Plan], frozen: FrozenCharacters, world_id: str, game_days: float | None, conv: _Conversation) -> None:
        with self._save_lock:
            for plan in plans:
                try:
                    self._inner.save_conversation_state(plan.thread, plan.npcs, frozen, world_id, False, None, game_days, False)
                except Exception as e:
                    logger.error(f"Memory checkpoint for {', '.join(plan.names)} failed ({e}); these messages will be saved later.")
                    unmark(plan.messages, plan.names)
        self._write_journal(conv)

    def wait_for_pending_checkpoints(self, timeout: float | None = None) -> None:
        for thread in list(self._threads):
            thread.join(timeout)

    # ---------- the normal saves (reload, NPC leaves, end of conversation) ----------
    def save_conversation_state(self, messages: Any, npcs_to_summarize: list[Any], npcs_in_conversation: Any, world_id: str,
                                is_reload=False, pending_shares: list[tuple[str, str, str]] | None = None,
                                end_timestamp: float | None = None, is_radiant: bool = False):
        if not is_reload:
            with self._state_lock:
                conv = self._conversation
                if self._is_current(conv, messages, npcs_in_conversation):
                    conv.ended = True
                    self._cancel_timer(conv)
        with self._save_lock:
            try:
                if is_radiant or not has_any_mark(messages):
                    self._inner.save_conversation_state(messages, npcs_to_summarize, npcs_in_conversation, world_id, is_reload,
                                                        pending_shares, end_timestamp, is_radiant)
                    if is_reload and not is_radiant:
                        mark_saved([messages[i] for i in range(len(messages)) if is_talk(messages[i])],
                                   [n.name for n in npcs_to_summarize if not getattr(n, "is_player_character", False)])
                    return
                for plan in plan_saves(messages, len(messages), npcs_to_summarize):
                    names = set(plan.names)
                    shares = [s for s in (pending_shares or []) if s[0] in names] or None
                    if not plan.messages and not shares:
                        continue
                    # After a checkpoint the rest may be short: use the lower reload threshold so it is still summarised
                    self._inner.save_conversation_state(plan.thread, plan.npcs, npcs_in_conversation, world_id,
                                                        is_reload or plan.had_checkpoint, shares, end_timestamp, False)
                    if is_reload:
                        mark_saved(plan.messages, plan.names)
            finally:
                if not is_reload:
                    self._finish(messages, npcs_in_conversation)

    @staticmethod
    def _is_current(conv: _Conversation | None, messages: Any, characters: Any) -> bool:
        return conv is not None and (conv.messages is messages or getattr(conv.context, "npcs_in_conversation", None) is characters)

    def _finish(self, messages: Any, characters: Any) -> None:
        with self._state_lock:
            conv = self._conversation
            if self._is_current(conv, messages, characters):
                self._conversation = None
                self._delete_journal(conv.conversation_id)

    # ---------- journal + recovery ----------
    def journal_path(self) -> str:
        return os.path.join(self.journal_folder, JOURNAL_NAME)

    def _journal_record(self, conv: _Conversation) -> dict[str, Any] | None:
        context = conv.context
        characters = context.npcs_in_conversation
        player_name = characters.get_player_name() or ""
        npcs = characters.get_non_player_characters()
        names = [n.name for n in npcs]
        messages = conv.messages
        pending = []
        for i in range(len(messages)):
            m = messages[i]
            if not is_talk(m) or all(is_saved_for(m, n) for n in names):
                continue
            content = m.get_formatted_content()
            speaker = _speaker(m, player_name)
            if not m.is_multi_npc_message and speaker:
                content = f"{speaker}: {content}"
            pending.append({"role": "user" if isinstance(m, UserMessage) else "assistant", "content": content})
        return {
            "schema_version": 1, "pid": os.getpid(), "conversation_id": conv.conversation_id,
            "world_id": context.world_id, "player_name": player_name, "game_days": getattr(context, "game_days", None),
            "npcs": [{"name": n.name, "ref_id": str(n.ref_id), "bio": n.bio or ""} for n in npcs],
            "nearby": list(characters.get_nearby_npc_names()), "checkpoints": conv.checkpoints,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pending": pending,
        }

    def _write_journal(self, conv: _Conversation) -> None:
        try:
            with self._state_lock:
                if conv.ended or conv is not self._conversation:
                    return
                record = self._journal_record(conv)
                os.makedirs(self.journal_folder, exist_ok=True)
                tmp = self.journal_path() + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.journal_path())
        except Exception as e:
            logger.error(f"Memory checkpoint: could not write the journal: {e}")

    def _delete_journal(self, conversation_id: str) -> None:
        path = self.journal_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    if json.load(f).get("conversation_id") != conversation_id:
                        return
                os.remove(path)
        except (OSError, ValueError) as e:
            logger.debug(f"Memory checkpoint: could not remove the journal: {e}")

    def recover_unfinished(self) -> bool:
        """Save the unsaved messages of a conversation that a server crash left behind. Returns True if saved."""
        path = self.journal_path()
        recovering = path + ".recovering"
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
                if record.get("pid") == os.getpid():
                    return False  # this process's live conversation, not a crashed one
                os.replace(path, recovering)
            if not os.path.exists(recovering):
                return False
            with open(recovering, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError) as e:
            logger.error(f"Memory checkpoint: unreadable journal ({e}); ignoring it.")
            try:
                os.replace(path, path + ".bak")
            except OSError:
                pass
            return False
        try:
            pending = [p for p in record.get("pending", []) if isinstance(p, dict) and p.get("content")]
            npcs = [RecoveredCharacter(str(n["name"]), str(n.get("ref_id", "")), str(n.get("bio", "")))
                    for n in record.get("npcs", []) if isinstance(n, dict) and n.get("name")]
            if npcs and len(pending) >= 2 and any(p.get("role") == "user" for p in pending):
                player_name = str(record.get("player_name") or "")
                thread = message_thread(None, None)
                for p in pending:
                    thread.add_message(RecoveredUserMessage(p["content"]) if p.get("role") == "user" else RecoveredAssistantMessage(p["content"]))
                player = [RecoveredCharacter(player_name, "", "", True)] if player_name else []
                log = [("join", c.name, 0) for c in player + npcs]
                frozen = FrozenCharacters(log, player + npcs, player_name or None, record.get("nearby") or [])
                logger.info(f"Memory checkpoint: saving {len(pending)} messages of the unfinished conversation "
                            f"{record.get('conversation_id')} with {', '.join(n.name for n in npcs)} (the server stopped before it ended).")
                with self._save_lock:
                    self._inner.save_conversation_state(thread, npcs, frozen, str(record.get("world_id") or "default"),
                                                        True, None, record.get("game_days"), False)
                saved = True
            else:
                saved = False
            os.remove(recovering)
            return saved
        except Exception as e:
            logger.error(f"Memory checkpoint: recovery failed: {e}")
            return False
