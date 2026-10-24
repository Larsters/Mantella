"""Persistent per-NPC relationship records (opinion of the player, facts, grudges, favours, promises).

Entry points:
- `create_rememberer(...)`: wraps the summaries rememberer when `relationship_tracking_enabled` is on.
- `prompt_text(...)` / `adjust_trust(...)`: used by Context to fill `{relationship}` and refine `{trust}`.
"""
from __future__ import annotations

from typing import Any

from src import utils

logger = utils.get_logger()


def create_rememberer(inner: Any, game: Any, config: Any, client: Any, language_name: str) -> Any:
    """Return `inner` unchanged unless relationship tracking is enabled in the config."""
    if not getattr(config, "relationship_tracking_enabled", False):
        return inner
    try:
        from src.relationships.tracker import RelationshipRememberer, RelationshipStore, RelationshipTracker
        from src.relationships.updater import DEFAULT_UPDATE_PROMPT
        store = RelationshipStore(game.conversation_folder_path)
        tracker = RelationshipTracker(
            store, client,
            game_name=getattr(game, "game_name_in_filepath", "Skyrim"),
            language=language_name,
            prompt_template=getattr(config, "relationship_update_prompt", "") or DEFAULT_UPDATE_PROMPT,
            token_budget=int(getattr(config, "relationship_token_budget", 150)),
            decay_per_day=float(getattr(config, "relationship_opinion_decay_per_day", 0)),
            affects_trust=bool(getattr(config, "relationship_affects_trust", True)),
            bystander_records=bool(getattr(config, "relationship_bystander_records", False)),
            resolve_ref_id=getattr(game, "resolve_npc_refid_by_name", None),
            lookup_bio=lambda name: _bio_from_character_db(game, name),
            track_lies=bool(getattr(config, "trust_and_honesty_enabled", False)),  # src/trust
        )
        logger.info("NPC relationship tracking enabled.")
        return RelationshipRememberer(inner, tracker)
    except Exception as e:
        logger.error(f"Could not enable NPC relationship tracking: {e}")
        return inner


def _bio_from_character_db(game: Any, name: str) -> str:
    """Bio of the single character-database entry with this name ("" if none or ambiguous)."""
    df = getattr(game, "character_df", None)
    if df is None or "name" not in getattr(df, "columns", []) or "bio" not in df.columns:
        return ""
    rows = df.loc[df["name"].astype(str).str.lower() == name.lower()]
    if rows.shape[0] != 1:
        return ""
    bio = rows.iloc[0].get("bio", "")
    return bio if isinstance(bio, str) else ""


def _tracker(rememberer: Any) -> Any:
    return getattr(rememberer, "relationship_tracker", None) if rememberer is not None else None


def prompt_text(rememberer: Any, npcs: list, world_id: str, player_name: str, current_day: float | None) -> str:
    """Text for the {relationship} placeholder ("" when the feature is off or nothing is known)."""
    tracker = _tracker(rememberer)
    if tracker is None:
        return ""
    try:
        return tracker.get_prompt_text(npcs, world_id, player_name, current_day)
    except Exception as e:
        logger.error(f"Could not load relationship records: {e}")
        return ""


def adjust_trust(rememberer: Any, base_trust: str, npc: Any, world_id: str, current_day: float | None) -> str:
    """Refine upstream's {trust} wording with the NPC's opinion (unchanged when the feature is off)."""
    tracker = _tracker(rememberer)
    if tracker is None:
        return base_trust
    return tracker.adjust_trust(base_trust, npc, world_id, current_day)
