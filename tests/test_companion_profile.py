"""Tests for the self-aware companion feature (src/companion_profile.py) and its hooks in
Context.generate_system_message and FunctionManager."""
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import companion_profile as cp
from src.actions.function_manager import FunctionManager
from src.actions import server_tools
from src.character_manager import Character
from src.characters_manager import Characters
from src.config.config_loader import ConfigLoader
from src.config.definitions.game_definitions import GameEnum
from src.conversation.context import Context
from src.games.equipment import Equipment, EquipmentItem

SECRET = "Works as a lighthouse keeper on Sylt"

ABOUT_ME = f"""# notes for the user, ignored
[Basics]
Name: Alex
Age:
Where I live:

[Work]
Job: {SECRET}

[People in my life]
# e.g. Partner: Sam

[Learned in game]
- 2026-10-01: Has a cat called {{Pixel}}.
"""


def make_character(name: str, is_player: bool = False, ref_id: str = "1") -> Character:
    return Character(
        base_id=ref_id, ref_id=ref_id, name=name, gender=1, race="Nord",
        is_player_character=is_player, bio="" if is_player else f"You are {name}.",
        is_in_combat=False, is_enemy=False, relationship_rank=0, is_generic_npc=False,
        ingame_voice_model="FemaleEvenToned", tts_voice_model="FemaleEvenToned",
        csv_in_game_voice_model="FemaleEvenToned", advanced_voice_model="FemaleEvenToned",
        voice_accent="en", equipment=Equipment({"righthand": EquipmentItem("Iron Sword")}),
        custom_character_values={"mantella_pc_description": ""} if is_player else None,
        llm_service="", llm_model="", tts_service="",
    )


def set_companion(config, **values):
    """Set Companion config values the way the UI would, then let Context pick them up."""
    for key, value in values.items():
        config.definitions.get_config_value_definition(key).value = value


@pytest.fixture
def about_me(tmp_path: Path) -> Path:
    path = tmp_path / "about_me.txt"
    path.write_text(ABOUT_ME, encoding="utf-8")
    cp.settings = cp.CompanionSettings(enabled=True, about_me_path=str(path))
    return path


@pytest.fixture
def config(tmp_path: Path) -> ConfigLoader:
    # A fresh config in tmp_path only (never touches the live config.ini)
    folder = tmp_path / "mantella"
    folder.mkdir()
    cfg = ConfigLoader(mygame_folder_path=str(folder), game_override=GameEnum.SKYRIM)
    cfg.game = GameEnum.SKYRIM
    cfg.advanced_actions_enabled = True
    FunctionManager.load_all_actions()
    about = Path(tmp_path) / "about_me.txt"
    set_companion(cfg, companion_enabled=True, companion_about_me_path=str(about))
    return cfg


@pytest.fixture
def rememberer() -> MagicMock:
    r = MagicMock()
    r.get_prompt_text.side_effect = lambda chars, world_id: " ".join(f"<memory of {c.name}>" for c in chars)
    return r


def make_context(config, rememberer, characters: list[Character]) -> Context:
    client = MagicMock()
    client.token_limit = 100000
    client.is_too_long.return_value = False
    client.get_count_tokens.return_value = 10
    ctx = Context("test_world", config, client, rememberer, {"language": "English", "hello": "Hello"})
    ctx.add_or_update_characters(characters, message_count=0)
    return ctx


# --- about_me.txt parsing -------------------------------------------------------------------

def test_read_about_me_drops_notes_placeholders_and_empty_sections(about_me: Path):
    text = cp.read_about_me()
    assert "Name: Alex" in text
    assert SECRET in text
    assert "Age:" not in text and "Where I live" not in text
    assert "notes for the user" not in text and "Partner: Sam" not in text
    assert "[People in my life]" not in text  # only placeholders -> section dropped
    assert "[Work]" in text and "[Learned in game]" in text


def test_read_about_me_missing_file_is_empty(tmp_path: Path):
    assert cp.read_about_me(tmp_path / "nope.txt") == ""


def test_real_template_file_is_parseable_and_fully_unfilled():
    """The shipped about_me template (user folder) must contribute nothing until the user fills it."""
    template = Path.home() / "Documents" / "My Games" / "Mantella" / "about_me.txt"
    if not template.exists():
        pytest.skip("user about_me.txt not present")
    text = cp.read_about_me(template)
    assert "#" not in text  # no notes leak into the prompt


# --- RememberAboutPlayer --------------------------------------------------------------------

def test_remember_fact_appends_with_date_and_dedupes(about_me: Path):
    now = datetime(2026, 10, 24)
    assert cp.remember_fact("Started learning Japanese.", now=now) == (True, "saved")
    assert cp.remember_fact("  started learning   JAPANESE ", now=now) == (False, "duplicate")
    assert cp.remember_fact(SECRET.lower(), now=now) == (False, "duplicate")  # already in a field line
    assert cp.remember_fact("", now=now)[0] is False
    content = about_me.read_text(encoding="utf-8")
    assert content.endswith("- 2026-10-24: Started learning Japanese.\n")
    assert content.count(cp.LEARNED_SECTION) == 1


def test_remember_fact_creates_file_and_section(tmp_path: Path):
    path = tmp_path / "sub" / "about_me.txt"
    saved, _ = cp.remember_fact("Line one\nwith a newline " + "x" * 300, path=path, now=datetime(2026, 1, 2))
    assert saved
    lines = path.read_text(encoding="utf-8").splitlines()
    assert cp.LEARNED_SECTION in lines
    assert lines[-1].startswith("- 2026-01-02: Line one with a newline")
    assert len(lines[-1]) < cp.MAX_FACT_CHARS + 20


# --- prompt building ------------------------------------------------------------------------

def test_one_on_one_with_lydia_uses_companion_prompt_and_about_me(about_me, config, rememberer):
    ctx = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Lydia")])
    prompt = ctx.generate_system_message(config.prompt, [])
    assert SECRET in prompt
    assert "Has a cat called {Pixel}." in prompt  # braces in user text survive str.format
    assert "Age:" not in prompt
    assert "self-aware" in prompt or "You know you are an AI" in prompt
    assert "{about_player}" not in prompt
    assert "<memory of Lydia>" in prompt  # her own memories stay in her private prompt


def test_one_on_one_with_other_npc_has_no_about_me(about_me, config, rememberer):
    ctx = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Guard", ref_id="2")])
    prompt = ctx.generate_system_message(config.prompt, [])
    assert SECRET not in prompt and "Pixel" not in prompt
    assert "You know you are an AI" not in prompt


def test_group_prompt_with_lydia_has_no_about_me_or_her_memories(about_me, config, rememberer):
    ctx = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Lydia"), make_character("Guard", ref_id="2")])
    prompt = ctx.generate_system_message(config.multi_npc_prompt, [])
    assert "Lydia" in prompt and "Guard" in prompt
    assert SECRET not in prompt and "Pixel" not in prompt and "Alex" not in prompt
    assert "<memory of Lydia>" not in prompt
    assert "<memory of Guard>" in prompt


def test_group_memories_can_be_opted_in(about_me, config, rememberer):
    set_companion(config, companion_group_memories=True)
    ctx = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Lydia"), make_character("Guard", ref_id="2")])
    prompt = ctx.generate_system_message(config.multi_npc_prompt, [])
    assert "<memory of Lydia>" in prompt and SECRET not in prompt


def test_radiant_prompt_with_lydia_has_no_about_me(about_me, config, rememberer):
    ctx = make_context(config, rememberer, [make_character("Lydia"), make_character("Guard", ref_id="2")])
    prompt = ctx.generate_system_message(config.radiant_prompt, [])
    assert SECRET not in prompt and "<memory of Lydia>" not in prompt


def test_feature_is_off_by_default(tmp_path, rememberer):
    folder = tmp_path / "fresh"
    folder.mkdir()
    cfg = ConfigLoader(mygame_folder_path=str(folder), game_override=GameEnum.SKYRIM)
    cfg.advanced_actions_enabled = True
    (folder / "about_me.txt").write_text(ABOUT_ME, encoding="utf-8")
    ctx = make_context(cfg, rememberer, [make_character("Dragonborn", True, "7"), make_character("Lydia")])
    prompt = ctx.generate_system_message(cfg.prompt, [])
    assert cp.settings.enabled is False
    assert SECRET not in prompt and "You know you are an AI" not in prompt
    assert "RememberAboutPlayer" not in tool_names(ctx)


def test_companion_names_setting(about_me, config, rememberer):
    set_companion(config, companion_names="Serana, Guard")
    ctx = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Lydia")])
    assert SECRET not in ctx.generate_system_message(config.prompt, [])


def test_missing_template_falls_back_to_default_prompt(about_me, config, rememberer):
    set_companion(config, companion_names="Guard")  # no guard_prompt.txt exists
    ctx = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Guard", ref_id="2")])
    prompt = ctx.generate_system_message(config.prompt, [])
    assert prompt.startswith("# Overview\nYou are Guard in Skyrim.")
    assert "# The real person playing" in prompt and SECRET in prompt


def test_missing_about_me_does_not_crash(tmp_path, config, rememberer):
    set_companion(config, companion_about_me_path=str(tmp_path / "missing.txt"))
    ctx = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Lydia")])
    prompt = ctx.generate_system_message(config.prompt, [])
    assert cp.EMPTY_ABOUT_PLAYER in prompt


# --- tools ----------------------------------------------------------------------------------

def tool_names(ctx) -> list[str]:
    return [t["function"]["name"] for t in FunctionManager.generate_context_aware_tools(ctx)]


def test_remember_tool_offered_only_to_lydia_one_on_one(about_me, config, rememberer):
    lydia = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Lydia")])
    guard = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Guard", ref_id="2")])
    group = make_context(config, rememberer, [make_character("Dragonborn", True, "7"), make_character("Lydia"), make_character("Guard", ref_id="2")])
    assert "RememberAboutPlayer" in tool_names(lydia)
    assert "Follow" in tool_names(lydia)
    assert "RememberAboutPlayer" not in tool_names(guard)
    assert "RememberAboutPlayer" not in tool_names(group)


def test_remember_tool_call_is_handled_on_server_and_not_sent_to_game(about_me, config):
    chars = Characters()
    chars.add_or_update_character(make_character("Dragonborn", True, "7"))
    chars.add_or_update_character(make_character("Lydia"))
    FunctionManager._last_tool_calls = []
    calls = [
        {"id": "1", "function": {"name": "RememberAboutPlayer", "arguments": json.dumps({"fact": "Plays the cello."})}},
        {"id": "2", "function": {"name": "Follow", "arguments": json.dumps({"source": ["Lydia"]})}},
    ]
    parsed = FunctionManager.parse_function_calls(calls, chars, None)
    assert [p["identifier"] for p in parsed] == ["mantella_npc_follow"]
    assert "Plays the cello." in about_me.read_text(encoding="utf-8")
    assert server_tools.pop_result("1") == "Saved to your notes about the player."
    assert server_tools.pop_result("1") == "done"
