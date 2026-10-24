from src.config.config_loader import ConfigLoader
from src.config.definitions.prompt_definitions import PromptDefinitions
from src.conversation.context import Context
from src.speech_flavor.presets import GERMAN_MIX


def test_speech_flavor_is_an_allowed_prompt_variable():
    assert "speech_flavor" in PromptDefinitions.ALLOWED_PROMPT_VARIABLES
    assert "speech_flavor" in PromptDefinitions.ALLOWED_PROMPT_VARIABLES_RADIANT
    checker = PromptDefinitions.PromptChecker(PromptDefinitions.ALLOWED_PROMPT_VARIABLES)
    assert checker.apply_constraint("# Rules\n{speech_flavor}\nRespond as {name}.").is_success


def test_config_defaults_keep_upstream_behaviour(default_config: ConfigLoader):
    assert default_config.speech_flavor == "None"
    assert default_config.speech_flavor_intensity == "Medium"
    assert default_config.speech_flavor_scope == "By race"
    assert default_config.speech_flavor_tts_respell is True


def test_placeholder_filled_in_system_message(default_config: ConfigLoader, default_context: Context):
    prompt = "You are {name}.\n# Rules\n{speech_flavor}\nStay in character."
    # default: empty
    assert "Speech style" not in default_context.generate_system_message(prompt, [])
    # enabled, scope 'All NPCs' (the fixture NPC is an Imperial guard)
    default_config.speech_flavor = "German mix"
    default_config.speech_flavor_scope = "All NPCs"
    default_config.speech_flavor_intensity = "Heavy"
    assert GERMAN_MIX.intensities["heavy"] in default_context.generate_system_message(prompt, [])
    # by race: Imperial gets one step lighter
    default_config.speech_flavor_scope = "By race"
    assert GERMAN_MIX.intensities["medium"] in default_context.generate_system_message(prompt, [])


def test_upstream_prompts_unchanged_when_disabled(default_config: ConfigLoader, default_context: Context):
    # prompts without the placeholder render exactly as before
    assert "Speech style" not in default_context.generate_system_message(default_config.prompt, [])
