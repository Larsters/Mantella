"""Glue between Mantella's conversation flow and the lore retriever.

Two injection channels (both gated by ``lore_enabled``, default off):

1. ``{lore}`` placeholder in the system prompt -> ``system_lore(context)``.
   Scene notes: the current location (and its hold), factions tied to that place, NPC/player race,
   factions named at the start of NPC bios. Computed whenever Mantella (re)builds the system prompt,
   i.e. at conversation start and when NPCs join/leave, so it is stable between turns and the
   system prompt stays a cacheable prefix.

2. Per-turn topical notes -> ``attach_turn_lore(context, user_message, thread)``.
   Notes about what the player just said (plus the last NPC line) are attached to that player's
   message only. The message is appended at the end of the thread and never changes afterwards,
   so earlier turns stay byte-identical and OpenAI's prefix cache keeps hitting. Because the note
   stays in the history, each note is injected at most once per conversation (repeating it would
   only cost tokens and anchor the NPC on the topic).

Notes are rendered into the LLM request only (UserMessage.get_openai_message); summaries and the
conversation memory use get_formatted_content / get_dict_formatted_string and never see them.
"""
from __future__ import annotations

import os
import weakref
from typing import Any

from src import utils
from src.lore.corpus import load_corpus
from src.lore.retriever import LoreQuery, LoreRetriever, format_notes

logger = utils.get_logger()

SYSTEM_HEADER = "Lore your character knows (background only; use it naturally, never recite it):"
TURN_PREFIX = "(Background lore the NPC may know, not spoken aloud: "
TURN_SUFFIX = ")"

_DEFAULT_FOLDER = os.path.join("data", "Skyrim", "lore")


class LoreSession:
    """Per-conversation lore state: retriever plus ids already placed in the prompt."""

    def __init__(self, retriever: LoreRetriever, max_notes: int, max_tokens: int, per_turn: bool):
        self.retriever = retriever
        self.max_notes = max_notes
        self.max_tokens = max_tokens
        self.per_turn = per_turn
        self.system_ids: set[str] = set()
        self.turn_ids: set[str] = set()

    def system_lore(self, query: LoreQuery) -> str:
        notes = self.retriever.scene_notes(query, self.max_notes, self.max_tokens)
        # Replaced, not accumulated: a rebuilt system prompt fully supersedes the previous one.
        self.system_ids = {n.id for n in notes}
        if notes:
            logger.log(23, f"Lore (system): {', '.join(n.id for n in notes)}")
        return format_notes(notes, SYSTEM_HEADER)

    def turn_lore(self, query: LoreQuery) -> str:
        if not self.per_turn or not query.utterance.strip():
            return ""
        notes = self.retriever.topic_notes(query, self.max_notes, self.max_tokens,
                                           exclude=self.system_ids | self.turn_ids)
        if not notes:
            return ""
        self.turn_ids.update(n.id for n in notes)
        logger.log(23, f"Lore (turn): {', '.join(n.id for n in notes)}")
        return TURN_PREFIX + " ".join(format_notes(notes, bullet="").splitlines()) + TURN_SUFFIX


# One session per Context object (a new Context is created for every conversation).
_sessions: "weakref.WeakKeyDictionary[Any, LoreSession | None]" = weakref.WeakKeyDictionary()


def _enabled(config) -> bool:
    if not getattr(config, "lore_enabled", False):
        return False
    try:
        from src.config.definitions.game_definitions import GameEnum
        return config.game.base_game == GameEnum.SKYRIM
    except Exception:
        return False


def lore_folders(config) -> list[str]:
    """Shipped notes first, then personal notes in the Mantella user folder (override by id)."""
    folders = [_DEFAULT_FOLDER]
    save_folder = getattr(config, "save_folder", None)
    if isinstance(save_folder, str) and save_folder:
        folders.append(os.path.join(save_folder, "data", "Skyrim", "lore"))
    return folders


def create_session(config) -> LoreSession:
    notes = load_corpus(lore_folders(config))
    retriever = LoreRetriever(notes, allow_spoilers=bool(getattr(config, "lore_allow_spoilers", False)))
    return LoreSession(
        retriever,
        max_notes=int(getattr(config, "lore_max_notes", 3)),
        max_tokens=int(getattr(config, "lore_max_tokens", 200)),
        per_turn=bool(getattr(config, "lore_per_turn", True)),
    )


def get_session(context) -> LoreSession | None:
    """Returns the lore session for this conversation, or None when the feature is off."""
    try:
        if context in _sessions:
            return _sessions[context]
        session = create_session(context.config) if _enabled(context.config) else None
        _sessions[context] = session
        return session
    except Exception as e:  # lore must never break a conversation
        logger.warning(f"Lore: disabled for this conversation after error: {e}")
        try:
            _sessions[context] = None
        except TypeError:
            pass
        return None


def build_query(context, utterance: str = "", previous_line: str = "") -> LoreQuery:
    chars = context.npcs_in_conversation
    npcs = chars.get_non_player_characters()
    player = chars.get_player_character()
    return LoreQuery(
        location=str(context.location or ""),
        utterance=utterance or "",
        previous_line=previous_line or "",
        npc_names=[n.name for n in npcs] + ([player.name] if player else []),
        npc_races=[str(n.race or "") for n in npcs],
        npc_bios=[str(n.bio or "") for n in npcs],
        player_race=str(player.race or "") if player else "",
    )


def system_lore(context) -> str:
    """Value for the ``{lore}`` prompt placeholder. Empty string when disabled or nothing matches."""
    session = get_session(context)
    if not session:
        return ""
    try:
        return session.system_lore(build_query(context))
    except Exception as e:
        logger.warning(f"Lore: system lore failed: {e}")
        return ""


def attach_turn_lore(context, user_message, thread) -> None:
    """Attaches topical notes for the player's latest words to ``user_message`` (LLM request only)."""
    session = get_session(context)
    if not session:
        return
    try:
        previous_line = ""
        last = thread.get_last_assistant_message() if thread is not None else None
        if last is not None:
            previous_line = last.get_formatted_content()
        text = session.turn_lore(build_query(context, user_message.text, previous_line))
        if text:
            user_message.lore_text = text
    except Exception as e:
        logger.warning(f"Lore: turn lore failed: {e}")
