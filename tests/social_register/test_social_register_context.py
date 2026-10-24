import pytest

from src.character_manager import Character
from src.config.config_loader import ConfigLoader
from src.conversation.context import Context
from src.prompt_profiles import get_profile_prompts
from src.social_register.data import STREET_TEXT
from src.speech_flavor.presets import GERMAN_MIX


def test_config_default_is_off(default_config: ConfigLoader):
    assert default_config.social_register_enabled is False


def _make_beggar(character: Character):
    character.name = "Brenuin"
    character.bio = "Brenuin is a Redguard man who lives in Whiterun. He is a homeless beggar and drunkard."
    character.race = "[Race <RedguardRace (00013748)>]"


def test_placeholder_carries_register(default_config: ConfigLoader, default_context: Context,
                                      example_skyrim_npc_character: Character):
    prompt = "You are {name}.\n# Rules\n{speech_flavor}\nStay in character."
    _make_beggar(example_skyrim_npc_character)
    assert "talks street" not in default_context.generate_system_message(prompt, [])
    default_config.social_register_enabled = True
    assert STREET_TEXT.format(names="Brenuin", s="s") in default_context.generate_system_message(prompt, [])


def test_fixture_guard_untouched(default_config: ConfigLoader, default_context: Context):
    default_config.social_register_enabled = True
    assert "street" not in default_context.generate_system_message("{speech_flavor}", [])


@pytest.mark.parametrize("which", ["prompt", "multi_npc_prompt", "radiant_prompt"])
def test_profile_prompts_format_with_all_placeholders(which, default_config: ConfigLoader, default_context: Context,
                                                      example_skyrim_npc_character: Character):
    _make_beggar(example_skyrim_npc_character)
    default_config.social_register_enabled = True
    default_config.speech_flavor = "German mix"
    default_config.speech_flavor_intensity = "Light"
    default_config.speech_flavor_scope = "All NPCs"
    prompt = getattr(get_profile_prompts("living_npcs_v1"), which)
    assert "{speech_flavor}" in prompt
    result = default_context.generate_system_message(prompt, [])
    assert "{" not in result and "}" not in result
    assert GERMAN_MIX.intensities["light"] in result
    assert "Brenuin talks street" in result
