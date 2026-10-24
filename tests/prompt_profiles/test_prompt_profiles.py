"""Every prompt (upstream defaults and every profile) must format with exactly the
variables Mantella passes, and pass the config UI's placeholder checker."""
import re
from string import Formatter

import pytest

from src.config.config_loader import ConfigLoader
from src.config.definitions.llm_definitions import NarrationHandlingEnum
from src.config.definitions.prompt_definitions import PromptDefinitions
from src.conversation.context import Context
from src.prompt_profiles import (
    DEFAULT_PROFILE, PROFILE_NAMES, apply_prompt_profile, get_profile_prompts,
)

# Keyword arguments Context.generate_system_message passes to str.format (keep in sync).
CONTEXT_FORMAT_KWARGS = {
    "player_name", "game", "player_description", "player_equipment", "name", "names", "names_w_player",
    "bio", "bios", "trust", "equipment", "location", "weather", "time", "current_day", "time_group",
    "language", "conversation_summary", "conversation_summaries", "actions", "knowledge_scope",
}

NARRATION_SETTINGS = [None, *list(NarrationHandlingEnum)]
NON_DEFAULT_PROFILES = [p for p in PROFILE_NAMES if p != DEFAULT_PROFILE]


def placeholders(text: str) -> set[str]:
    return {field for _, field, _, _ in Formatter().parse(text) if field}


def test_context_passes_every_allowed_variable():
    """Any allowed variable the context did not pass would crash prompt.format()."""
    assert set(PromptDefinitions.ALLOWED_PROMPT_VARIABLES) <= CONTEXT_FORMAT_KWARGS
    assert set(PromptDefinitions.ALLOWED_PROMPT_VARIABLES_RADIANT) <= CONTEXT_FORMAT_KWARGS


def test_context_source_passes_the_documented_kwargs():
    import inspect
    source = inspect.getsource(inspect.getmodule(Context))
    source = source[source.index("def generate_system_message"):]
    call = source[source.index("prompt.format("):source.index("is_too_long")]
    passed = set(re.findall(r"^\s*(\w+)\s*=", call, re.MULTILINE))
    assert CONTEXT_FORMAT_KWARGS <= passed


@pytest.mark.parametrize("profile_name", NON_DEFAULT_PROFILES)
@pytest.mark.parametrize("narration", NARRATION_SETTINGS)
def test_profile_prompts_pass_ui_checker_and_format(profile_name, narration):
    profile = get_profile_prompts(profile_name, narration)
    checks = [
        (profile.prompt, PromptDefinitions.ALLOWED_PROMPT_VARIABLES),
        (profile.multi_npc_prompt, PromptDefinitions.ALLOWED_PROMPT_VARIABLES),
        (profile.radiant_prompt, PromptDefinitions.ALLOWED_PROMPT_VARIABLES_RADIANT),
        (profile.radiant_start_prompt, []),
        (profile.radiant_end_prompt, []),
    ]
    dummy = {k: f"<{k}>" for k in CONTEXT_FORMAT_KWARGS}
    for text, allowed in checks:
        result = PromptDefinitions.PromptChecker(allowed).apply_constraint(text)
        assert result.is_success, result.error_message
        assert placeholders(text) <= set(allowed)
        formatted = text.format(**dummy)
        assert "{" not in formatted and "}" not in formatted
        assert "voice_narration_rule" not in formatted


@pytest.mark.parametrize("profile_name", NON_DEFAULT_PROFILES)
def test_profile_rules_are_static_prefix(profile_name):
    """Rules come before any placeholder so they are identical for every NPC (prompt cache)."""
    profile = get_profile_prompts(profile_name)
    for text in (profile.prompt, profile.multi_npc_prompt, profile.radiant_prompt):
        first_placeholder = text.index("{")
        assert first_placeholder > text.index("# Who") > 1000
        assert "# Right now" in text[first_placeholder:]


@pytest.mark.parametrize("profile_name", NON_DEFAULT_PROFILES)
def test_single_prompt_forbids_colons_and_uses_trust(profile_name):
    profile = get_profile_prompts(profile_name)
    assert "Never use a colon" in profile.prompt
    assert "{trust}" in profile.prompt and "{trust}" in profile.multi_npc_prompt
    assert "{knowledge_scope}" in profile.prompt
    # The bio may override the fourth-wall rule (opt-in self-aware characters)
    assert "Unless your background says otherwise" in profile.prompt


def test_narration_rule_follows_setting():
    cut = get_profile_prompts("living_npcs_v1", NarrationHandlingEnum.CUT_NARRATIONS).prompt
    narrator = get_profile_prompts("living_npcs_v1", NarrationHandlingEnum.USE_NARRATOR).prompt
    assert "No narration" in cut and "between asterisks" not in cut
    assert "between asterisks" in narrator


def test_upstream_default_prompts_format(default_config: ConfigLoader):
    dummy = {k: f"<{k}>" for k in CONTEXT_FORMAT_KWARGS}
    for text in (default_config.prompt, default_config.multi_npc_prompt, default_config.radiant_prompt):
        text.format(**dummy)


def test_default_profile_keeps_config_prompts(default_config: ConfigLoader):
    assert default_config.prompt_profile == DEFAULT_PROFILE
    assert default_config.knowledge_scope_enabled is True
    assert default_config.guided_action_descriptions is False
    before = default_config.prompt
    apply_prompt_profile(default_config)
    assert default_config.prompt == before
    assert get_profile_prompts("does-not-exist") is None


def test_apply_profile_replaces_skyrim_prompts(default_config: ConfigLoader):
    default_config.prompt_profile = "living_npcs_v1"
    apply_prompt_profile(default_config)
    profile = get_profile_prompts("living_npcs_v1", default_config.narration_handling)
    assert default_config.prompt == profile.prompt
    assert default_config.multi_npc_prompt == profile.multi_npc_prompt
    assert default_config.radiant_prompt == profile.radiant_prompt
    assert default_config.radiant_start_prompt == profile.radiant_start_prompt
    assert default_config.radiant_end_prompt == profile.radiant_end_prompt


def test_profile_loads_from_config_ini(tmp_path):
    (tmp_path / "config.ini").write_text("[Prompts]\nprompt_profile = living_npcs_v1\n", encoding="utf-8")
    config = ConfigLoader(mygame_folder_path=str(tmp_path))
    assert config.prompt_profile == "living_npcs_v1"
    assert config.prompt.startswith("# How to play your part")


def test_context_renders_profile_prompt_with_knowledge_scope(default_config: ConfigLoader, default_context: Context):
    profile = get_profile_prompts("living_npcs_v1")
    result = default_context.generate_system_message(profile.prompt, [])
    assert "You are Guard." in result
    assert "Guard is an Imperial guard" in result      # knowledge scope rendered
    assert "Dragonborn is a stranger to Guard." in result  # trust rendered
    assert "{" not in result


def test_context_knowledge_scope_disabled(default_config: ConfigLoader, default_context: Context):
    default_config.knowledge_scope_enabled = False
    profile = get_profile_prompts("living_npcs_v1")
    result = default_context.generate_system_message(profile.prompt, [])
    assert "Guard is an Imperial guard" not in result


def test_context_formats_game_placeholder(default_context: Context):
    assert default_context.generate_system_message("Game {game}.", []) == "Game Skyrim."
