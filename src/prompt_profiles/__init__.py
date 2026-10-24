"""Selectable prompt profiles.

A profile is a named, versioned set of Skyrim prompts (single NPC, multi NPC, radiant,
radiant start/end). The default profile ("default") changes nothing: Mantella uses the
prompts from config.ini / the web UI exactly as upstream does. Any other profile replaces
those prompts at config load time, so the profile name alone identifies the prompt text
(useful for A/B evaluations: record `config.prompt_profile` with every result).

To try a profile's text but edit it yourself, copy it from `get_profile_prompts()` (or the
feature doc) into the Prompts tab and set the profile back to "default".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.prompt_profiles import living_npcs_v1

if TYPE_CHECKING:
    from src.config.config_loader import ConfigLoader

DEFAULT_PROFILE = "default"


@dataclass(frozen=True)
class PromptProfile:
    prompt: str
    multi_npc_prompt: str
    radiant_prompt: str
    radiant_start_prompt: str
    radiant_end_prompt: str


def _narration_rule(narration_handling) -> str:
    """Voice rule matching the narration setting (see living_npcs_v1)."""
    try:
        from src.config.definitions.llm_definitions import NarrationHandlingEnum
        if narration_handling in (NarrationHandlingEnum.RESPECTIVE_CHARACTER_SPEAKS_NARRATION,
                                  NarrationHandlingEnum.USE_NARRATOR):
            return living_npcs_v1.VOICE_WITH_NARRATION
    except Exception:
        pass
    return living_npcs_v1.VOICE_NO_NARRATION


def _living_npcs_v1(narration_handling) -> PromptProfile:
    rule = _narration_rule(narration_handling)
    m = living_npcs_v1
    return PromptProfile(
        prompt=m.SINGLE_RULES.replace("{voice_narration_rule}", rule) + m.SINGLE_CONTEXT,
        multi_npc_prompt=m.MULTI_RULES.replace("{voice_narration_rule}", rule) + m.MULTI_CONTEXT,
        radiant_prompt=m.RADIANT_RULES.replace("{voice_narration_rule}", rule) + m.RADIANT_CONTEXT,
        radiant_start_prompt=m.RADIANT_START,
        radiant_end_prompt=m.RADIANT_END,
    )


_PROFILES = {
    "living_npcs_v1": _living_npcs_v1,
}

PROFILE_NAMES: list[str] = [DEFAULT_PROFILE, *_PROFILES.keys()]


def get_profile_prompts(profile_name: str, narration_handling=None) -> PromptProfile | None:
    """Returns the prompts of a profile, or None for "default" / unknown names."""
    builder = _PROFILES.get((profile_name or "").strip())
    return builder(narration_handling) if builder else None


def apply_prompt_profile(config: "ConfigLoader") -> None:
    """Replaces the Skyrim prompts on a loaded config with the selected profile's prompts.
    No-op for the default profile, unknown profiles and non-Skyrim games."""
    profile_name = getattr(config, "prompt_profile", DEFAULT_PROFILE)
    profile = get_profile_prompts(profile_name, getattr(config, "narration_handling", None))
    if profile is None:
        return
    try:
        from src.config.definitions.game_definitions import GameEnum
        if config.game.base_game != GameEnum.SKYRIM:
            return
    except Exception:
        return
    config.prompt = profile.prompt
    config.multi_npc_prompt = profile.multi_npc_prompt
    config.radiant_prompt = profile.radiant_prompt
    config.radiant_start_prompt = profile.radiant_start_prompt
    config.radiant_end_prompt = profile.radiant_end_prompt
