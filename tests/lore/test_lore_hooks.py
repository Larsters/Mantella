"""Integration of src/lore with Context / UserMessage (no LLM calls)."""
from src.config.config_loader import ConfigLoader
from src.conversation.context import Context
from src.llm.message_thread import message_thread
from src.llm.messages import UserMessage
from src.lore import lore_hooks

PROMPT = "You are {name} in {location}.\n{lore}\nEnd."


def _enable(config: ConfigLoader, **overrides):
    config.lore_enabled = True
    config.lore_per_turn = True
    config.lore_max_notes = 3
    config.lore_max_tokens = 200
    config.lore_allow_spoilers = False
    for k, v in overrides.items():
        setattr(config, k, v)


def _at(context: Context, location: str):
    context.update_context(location, 12, None, None, None, {}, None)


def test_disabled_by_default_is_upstream_behaviour(default_config: ConfigLoader, default_context: Context):
    assert default_config.lore_enabled is False
    _at(default_context, "Markarth")
    prompt = default_context.generate_system_message(PROMPT, [])
    assert "Lore" not in prompt and "Forsworn" not in prompt
    msg = UserMessage(default_config, "What do you think about the war?", "Dragonborn")
    lore_hooks.attach_turn_lore(default_context, msg, message_thread(default_config, "sys"))
    assert msg.lore_text == ""
    assert msg.get_openai_message()["content"] == msg.get_formatted_content()


def test_system_prompt_gets_scene_lore(default_config: ConfigLoader, default_context: Context):
    _enable(default_config)
    _at(default_context, "Markarth")
    prompt = default_context.generate_system_message(PROMPT, [])
    assert lore_hooks.SYSTEM_HEADER in prompt
    assert "Markarth:" in prompt and "The Forsworn:" in prompt and "Silver-Blood family" in prompt
    # stable between rebuilds -> cache-friendly
    assert default_context.generate_system_message(PROMPT, []) == prompt


def test_prompt_without_placeholder_is_unchanged(default_config: ConfigLoader, default_context: Context):
    _enable(default_config)
    _at(default_context, "Markarth")
    assert "Forsworn" not in default_context.generate_system_message("You are {name} in {location}.", [])


def test_turn_lore_attached_once_and_kept_out_of_summaries(default_config: ConfigLoader, default_context: Context):
    _enable(default_config)
    _at(default_context, "Whiterun")
    default_context.generate_system_message(PROMPT, [])  # scene notes (whiterun ...) are excluded from turns
    thread = message_thread(default_config, "sys")

    first = UserMessage(default_config, "What do you think about the war?", "Dragonborn")
    lore_hooks.attach_turn_lore(default_context, first, thread)
    assert "civil war" in first.lore_text.lower()
    assert "Whiterun: Whiterun is" not in first.lore_text
    content = first.get_openai_message()["content"]
    assert content.startswith(lore_hooks.TURN_PREFIX) and content.endswith("What do you think about the war?")
    assert lore_hooks.TURN_PREFIX not in first.get_formatted_content()
    assert lore_hooks.TURN_PREFIX not in first.get_dict_formatted_string()

    second = UserMessage(default_config, "And what about the war again?", "Dragonborn")
    lore_hooks.attach_turn_lore(default_context, second, thread)
    assert "The Skyrim civil war:" not in second.lore_text  # not repeated within a conversation


def test_new_conversation_gets_fresh_session(default_config: ConfigLoader, default_context: Context, llm_client, default_rememberer, english_language_info):
    _enable(default_config)
    s1 = lore_hooks.get_session(default_context)
    other = Context('2', default_config, llm_client, default_rememberer, english_language_info)
    assert lore_hooks.get_session(other) is not s1


def test_fallout_is_ignored(default_config: ConfigLoader, default_context: Context, monkeypatch):
    from src.config.definitions.game_definitions import GameEnum
    _enable(default_config)
    monkeypatch.setattr(default_config, "game", GameEnum.FALLOUT4, raising=False)
    fresh = Context('3', default_config, default_context._Context__client, default_context._Context__rememberer, default_context.language)
    assert lore_hooks.get_session(fresh) is None
