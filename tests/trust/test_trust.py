"""Trust and honesty (src/trust): prompt rules, tool-model prompt + scene, tool descriptions,
and lies/credibility in the W6 relationship record. No network."""
import json
from string import Formatter
from types import SimpleNamespace

import pytest

from src import trust
from src.relationships import record as rec
from src.relationships import render, updater
from src.relationships.tracker import RelationshipStore, RelationshipTracker
from src.trust import rules
from src.prompt_profiles import get_profile_prompts
from src.config.definitions.prompt_definitions import PromptDefinitions
from tests.relationships.test_tracker import FakeClient, _conversation, _npc
from tests.prompt_profiles.test_prompt_profiles import CONTEXT_FORMAT_KWARGS


# ---------- record: migration, lies, credibility ----------
def test_v1_record_migrates_with_neutral_defaults_and_renders_as_before():
    v1 = {"schema_version": 1, "npc_name": "Guard", "opinion": 10, "attitude": "Friendly enough.",
          "conversation_count": 2, "facts": ["Is a bard"], "grudges": [], "history": []}
    record = rec.normalize_record(v1, "Guard")
    assert record["credibility"] == 50 and record["lies"] == []
    text = render.render_record(record, "Eirik", 5, 150)
    assert "lying" not in text and "word" not in text  # nothing new for the neutral default
    # unknown / broken values are repaired
    assert rec.normalize_record({"credibility": "x", "lies": "not a list"})["credibility"] == 50
    assert rec.normalize_record({"credibility": 500})["credibility"] == 100


def test_caught_lie_is_recorded_and_costs_credibility_and_opinion():
    record = rec.new_record("Whiterun Guard", "1")
    update = {"opinion_change": 0, "reason": "", "credibility_change": -5,  # model too lenient: floors apply
              "caught_lies": [{"text": "Claimed a bomb was hidden in the barrels; there was nothing", "witnesses": ["Adrianne"]}]}
    new = rec.apply_update(record, update, 12)
    assert new["schema_version"] == 2
    assert new["lies"] == [{"id": "l1", "text": "Claimed a bomb was hidden in the barrels; there was nothing", "day": 12, "witnesses": ["Adrianne"]}]
    assert new["credibility"] == 50 - rec.MIN_CREDIBILITY_DROP_PER_LIE
    assert new["opinion"] == -rec.MIN_OPINION_DROP_PER_LIE
    assert any(h["reason"].startswith("Caught lying") for h in new["history"])

    # The same lie reported again is deduplicated and not punished twice
    again = rec.apply_update(new, {"caught_lies": ["Claimed a bomb was hidden in the barrels, there was nothing"]}, 13)
    assert len(again["lies"]) == 1 and again["credibility"] == new["credibility"]

    # A second, different lie: credibility drops into "believes nothing"
    twice = rec.apply_update(again, {"caught_lies": ["Swore the Jarl had sent him"], "credibility_change": -25}, 14)
    assert len(twice["lies"]) == 2 and twice["credibility"] <= 20


def test_credibility_gain_is_bounded_and_capped_while_lies_stand():
    record = rec.new_record("Lydia")
    assert rec.apply_update(record, {"credibility_change": 99}, 1)["credibility"] == 50 + rec.MAX_CREDIBILITY_GAIN
    liar = rec.apply_update(record, {"caught_lies": ["a", "b lie that is different", "third invented story"]}, 1)
    for day in range(2, 12):
        liar = rec.apply_update(liar, {"credibility_change": 15}, day)
    assert liar["credibility"] == rec.credibility_ceiling(3) == 40
    loss = rec.apply_update(rec.new_record("X"), {"credibility_change": -500}, 1)
    assert loss["credibility"] == 50 - rec.MAX_CREDIBILITY_LOSS


def test_lies_are_capped_and_bystanders_get_a_third_of_the_penalty():
    record = rec.new_record("X")
    for i in range(10):
        record = rec.apply_update(record, {"caught_lies": [f"Lie number {i} about completely unrelated topic {i * 7}"]}, i)
    assert len(record["lies"]) == rec.CAPS["lies"]
    bystander = rec.apply_update(rec.new_record("Alfhild"), {"caught_lies": ["Lied to Lydia about the dragon"], "credibility_change": -40}, 3, bystander=True)
    assert bystander["lies"][0]["overheard"] is True
    assert bystander["credibility"] == 50 - rec.MAX_BYSTANDER_CREDIBILITY_CHANGE
    assert bystander["opinion"] == -2  # -5 // 3


def test_render_shows_lies_and_credibility_first():
    record = rec.new_record("Whiterun Guard")
    record = rec.apply_update(record, {"caught_lies": ["Claimed a bomb was in the barrels"],
                                       "new_grudges": [{"text": "Wasted my time", "severity": 1}]}, 10)
    record = rec.apply_update(record, {"caught_lies": ["Said Balgruuf sent him"], "credibility_change": -20}, 11)
    text = render.render_record(record, "Eirik", 12, 150)
    assert "Whiterun Guard believes nothing Eirik claims without proof" in text
    assert "has caught Eirik lying twice, most recently: Said Balgruuf sent him (yesterday)." in text
    assert text.index("lying twice") < text.index("Grudge:")
    # with a tiny budget the credibility sentence survives (part of the opinion line)
    assert "believes nothing" in render.render_record(record, "Eirik", 12, 10)
    honest = rec.normalize_record({"credibility": 80, "conversation_count": 1}, "Lydia")
    assert "has found Eirik honest" in render.render_record(honest, "Eirik", 1, 150)


# ---------- update call ----------
def test_update_schema_and_messages_only_change_when_lies_are_tracked():
    assert updater.update_json_schema(False) is updater.UPDATE_JSON_SCHEMA
    schema = updater.update_json_schema(True)
    item = schema["properties"]["npcs"]["items"]
    assert {"caught_lies", "credibility_change"} <= set(item["required"]) <= set(item["properties"])
    assert "caught_lies" not in updater.UPDATE_JSON_SCHEMA["properties"]["npcs"]["items"]["properties"]  # not mutated
    npcs = [("Guard", "A guard.", rec.new_record("Guard"))]
    _, plain = updater.build_messages(updater.DEFAULT_UPDATE_PROMPT, "Skyrim", "English", "Eirik", npcs, [], "t", 3)
    _, lies = updater.build_messages(updater.DEFAULT_UPDATE_PROMPT, "Skyrim", "English", "Eirik", npcs, [], "t", 3, track_lies=True)
    assert "caught_lies" not in plain and '"credibility"' not in plain
    assert "caught_lies" in lies and '"credibility": 50' in lies and "{player_name}" not in lies


def test_tracker_records_a_caught_lie_from_the_update_call(tmp_path):
    people = {"player": _npc("Dragonborn", "14", is_player=True), "lydia": _npc("Lydia", "A2C94", "Housecarl."),
              "guard": _npc("Guard", "0", "A guard.")}
    thread, _ = _conversation(people)
    reply = json.dumps({"npcs": [
        {"name": "Guard", "opinion_change": -10, "reason": "lied to me", "attitude": "Suspicious.", "new_facts": [],
         "new_grudges": [], "resolved_grudges": [], "new_favours": [], "settled_favours": [], "new_promises": [],
         "resolved_promises": [], "caught_lies": [{"text": "Claimed a grenade was in the barrel", "witnesses": []}],
         "credibility_change": -25}]})
    client = FakeClient([reply])
    tracker = RelationshipTracker(RelationshipStore(str(tmp_path)), client, "Skyrim", "English",
                                  run_in_background=False, track_lies=True)
    tracker.update_records([people["guard"]], "Dragonborn: there is a grenade in the barrel", "w1", "Dragonborn", [], 12)
    params = client.calls[0]["params"]
    assert params["response_format"] == updater.response_format(True)
    assert "caught_lies" in client.calls[0]["messages"][1]["content"]
    saved = tracker.store.load("Guard", "0", "w1")
    assert saved["lies"][0]["text"] == "Claimed a grenade was in the barrel"
    assert saved["credibility"] == 25 and saved["opinion"] == -10
    with open(tracker.store.path_for("Guard", "0", "w1")) as f:
        assert json.load(f)["schema_version"] == 2


def test_create_rememberer_passes_track_lies(default_config, tmp_path):
    from src import relationships
    game = SimpleNamespace(conversation_folder_path=str(tmp_path), game_name_in_filepath="Skyrim", character_df=None)
    default_config.relationship_tracking_enabled = True
    default_config.trust_and_honesty_enabled = True
    wrapped = relationships.create_rememberer(SimpleNamespace(), game, default_config, FakeClient([]), "English")
    assert wrapped.relationship_tracker.track_lies is True
    default_config.trust_and_honesty_enabled = False
    wrapped = relationships.create_rememberer(SimpleNamespace(), game, default_config, FakeClient([]), "English")
    assert wrapped.relationship_tracker.track_lies is False


# ---------- prompts ----------
def _cfg(**kw):
    return SimpleNamespace(trust_and_honesty_enabled=True, trust_tool_model_context=True, **kw)


def test_config_default_is_off(default_config):
    assert default_config.trust_and_honesty_enabled is False
    assert default_config.trust_tool_model_context is True
    assert rules.SINGLE_BLOCK not in default_config.prompt


def test_rules_have_no_placeholders():
    for block in (rules.SINGLE_BLOCK, rules.MULTI_BLOCK, rules.FUNCTION_BLOCK):
        assert not [f for _, f, _, _ in Formatter().parse(block) if f]


def test_living_npcs_prompts_get_the_block_and_still_format():
    profile = get_profile_prompts("living_npcs_v1")
    cfg = _cfg(prompt=profile.prompt, multi_npc_prompt=profile.multi_npc_prompt)
    trust.apply_prompt_rules(cfg)
    trust.apply_prompt_rules(cfg)  # idempotent
    for prompt, block, marker in ((cfg.prompt, rules.SINGLE_BLOCK, "\n\nCharacter\n"), (cfg.multi_npc_prompt, rules.MULTI_BLOCK, "\n\nCharacters\n")):
        assert prompt.count(block) == 1
        assert prompt.index("Actions\n") < prompt.index(block) < prompt.index(marker)
        assert prompt.index(block) < prompt.index("{")  # static rules stay in the cacheable prefix
        filled = prompt.format(**{k: f"<{k}>" for k in CONTEXT_FORMAT_KWARGS})
        assert "Requests, claims and lies" in filled
        assert PromptDefinitions.PromptChecker(PromptDefinitions.ALLOWED_PROMPT_VARIABLES).apply_constraint(prompt).is_success


def test_custom_prompt_gets_block_in_front_and_off_changes_nothing():
    assert trust.insert_block("You are {name}.", "RULES", "\n\nCharacter\n") == "RULES\n\nYou are {name}."
    cfg = SimpleNamespace(trust_and_honesty_enabled=False, prompt="P", multi_npc_prompt="M")
    trust.apply_prompt_rules(cfg)
    assert (cfg.prompt, cfg.multi_npc_prompt) == ("P", "M")


# ---------- tool model ----------
def test_function_prompt_contains_rules_and_scene():
    base = "You are analyzing a conversation in Skyrim."
    assert trust.function_prompt(SimpleNamespace(trust_and_honesty_enabled=False), base) == base
    prompt = trust.function_prompt(_cfg(), base)
    assert prompt.startswith(base) and rules.FUNCTION_BLOCK in prompt
    for must in ("means Look", "CheckDirections only for the way to a named place", "ReportCrime only charges the player",
                 "bombs, grenades", "checks first", "caught lying"):
        assert must in prompt
    guard = _npc("Whiterun Guard", "1", "A guard of Whiterun. " + "x " * 300)
    trust.publish_scene(_cfg(), [guard, _npc("Eirik", "14", is_player=True)], "Eirik", "a stranger to Whiterun Guard",
                        "Relationship with the player:\nWhiterun Guard doubts Eirik's word.")
    full = trust.function_system_prompt(_cfg(), prompt)
    assert full.startswith(prompt)
    scene = full[len(prompt):]
    assert "- Whiterun Guard: A guard of Whiterun." in scene and "Eirik is a stranger to Whiterun Guard." in scene
    assert "doubts Eirik's word" in scene and "- Eirik" not in scene and len(scene) < 600
    no_scene = SimpleNamespace(trust_and_honesty_enabled=True, trust_tool_model_context=False)
    assert trust.function_system_prompt(no_scene, prompt) == prompt
    trust.clear_scene()
    assert trust.function_system_prompt(_cfg(), prompt) == prompt


def test_function_client_uses_trust_prompt(default_config):
    from src.actions.function_manager import FunctionManager
    from src.llm.function_client import FunctionClient
    FunctionManager.load_all_actions()
    default_config.trust_and_honesty_enabled = True
    client = FunctionClient(default_config)
    assert rules.FUNCTION_BLOCK in client._FunctionClient__function_prompt
    default_config.trust_and_honesty_enabled = False
    assert rules.FUNCTION_BLOCK not in FunctionClient(default_config)._FunctionClient__function_prompt


def test_tool_description_precedence():
    action = {"name": "Loot", "description": "up", "guided_description": "guided", "trust_description": "trust"}
    assert trust.tool_description(action, SimpleNamespace(guided_action_descriptions=False)) == "up"
    assert trust.tool_description(action, SimpleNamespace(guided_action_descriptions=True)) == "guided"
    assert trust.tool_description(action, SimpleNamespace(guided_action_descriptions=False, trust_and_honesty_enabled=True)) == "trust"
    no_trust = {"name": "Attack", "description": "up", "guided_description": "guided"}
    assert trust.tool_description(no_trust, SimpleNamespace(guided_action_descriptions=True, trust_and_honesty_enabled=True)) == "guided"


def test_tools_sent_with_trust_descriptions(default_config, default_context):
    from src.actions.function_manager import FunctionManager
    FunctionManager.load_all_actions([])
    default_config.guided_action_descriptions = True
    default_config.trust_and_honesty_enabled = True
    tools = {t["function"]["name"]: t["function"]["description"] for t in FunctionManager.generate_context_aware_tools(default_context)}
    assert "check" in tools["Look"] and "barrel" in tools["Look"]
    assert "Never for objects or people nearby" in tools["CheckDirections"]
    assert "never for claims about things that do not exist in Skyrim" in tools["ReportCrime"]
    assert "seriously threatens to kill" in tools["Attack"]  # untouched tools keep the guided text
    default_config.trust_and_honesty_enabled = False
    tools = {t["function"]["name"]: t["function"]["description"] for t in FunctionManager.generate_context_aware_tools(default_context)}
    assert "Never for objects" not in tools["CheckDirections"]


def test_context_publishes_scene_for_tool_model(default_config, default_rememberer, llm_client, english_language_info, example_characters_pc_to_npc):
    from src.conversation.context import Context
    trust.clear_scene()
    default_config.trust_and_honesty_enabled = True
    context = Context("1", default_config, llm_client, default_rememberer, english_language_info)
    context.add_or_update_characters(example_characters_pc_to_npc.get_all_characters(), message_count=0)
    context.generate_system_message("{name} {trust}", [])
    assert "- Guard" in trust.scene_text() and "is a stranger to Guard." in trust.scene_text()
    trust.clear_scene()
    default_config.trust_and_honesty_enabled = False
    context.generate_system_message("{name} {trust}", [])
    assert trust.scene_text() == ""


def test_trust_descriptions_are_lean():
    from pathlib import Path
    for path in (Path(__file__).resolve().parents[2] / "data" / "actions").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "trust_description" in data:
            assert len(data["trust_description"].split()) <= 75, path.name
