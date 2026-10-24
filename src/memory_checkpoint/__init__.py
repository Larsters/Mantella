"""Crash-safe memory (F3): checkpoint summaries and relationship updates during a conversation.

Entry points (all no-ops unless `memory_checkpoint_enabled` is on):
- `create_rememberer(inner, game, config)`  game_manager: wraps the rememberer, starts startup recovery
- `on_player_turn(rememberer, messages, context)`  Conversation: after each normal player line
"""
from __future__ import annotations

import os
import threading
from typing import Any

from src import utils

logger = utils.get_logger()


def journal_folder(game: Any) -> str:
    """`data/<game>/memory_checkpoints`, next to the `conversations` folder."""
    return os.path.join(os.path.dirname(os.path.normpath(game.conversation_folder_path)), "memory_checkpoints")


def create_rememberer(inner: Any, game: Any, config: Any, recover_in_background: bool = True) -> Any:
    """Return `inner` unchanged unless memory checkpoints are enabled in the config."""
    if not getattr(config, "memory_checkpoint_enabled", False):
        return inner
    try:
        from src.memory_checkpoint.checkpointer import CheckpointRememberer
        wrapper = CheckpointRememberer(inner, journal_folder(game),
                                       every_n_turns=int(getattr(config, "memory_checkpoint_every_n_turns", 6)),
                                       idle_minutes=float(getattr(config, "memory_checkpoint_idle_minutes", 3)))
        # Message marks replace W6's content-based reload dedupe (it would drop repeated lines like "Yes.")
        tracker = getattr(inner, "relationship_tracker", None)
        if tracker is not None:
            tracker.dedupe_reloads = False
        logger.info(f"Memory checkpoints enabled (every {wrapper.every_n_turns} player turns, "
                    f"after {getattr(config, 'memory_checkpoint_idle_minutes', 3)} idle minutes).")
        if getattr(config, "memory_checkpoint_recover_on_startup", True):
            if recover_in_background:
                threading.Thread(target=wrapper.recover_unfinished, name="memory-checkpoint-recovery", daemon=False).start()
            else:
                wrapper.recover_unfinished()
        return wrapper
    except Exception as e:
        logger.error(f"Could not enable memory checkpoints: {e}")
        return inner


def on_player_turn(rememberer: Any, messages: Any, context: Any) -> None:
    from src.memory_checkpoint.checkpointer import CheckpointRememberer
    if isinstance(rememberer, CheckpointRememberer):
        rememberer.on_player_turn(messages, context)
