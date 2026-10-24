"""Tests for src/bio_generation (W7). The LLM is always mocked; no network calls."""
import json
import os
import threading
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.bio_generation import (BioGenerator, BioQuality, BioScope, BioStore, TemplateIndex, classify,
                                create_bio_generator, location_from_request, make_llm_call)
from src.bio_generation.npc_facts import parse_race, parse_voice_type, variety_seed, voice_hint, hold_for, is_creature_race
from src.bio_generation.prompt import build_messages, parse_response
from src.bio_generation.npc_facts import facts_from_character
from src.character_manager import Character
from src.games.equipment import Equipment, EquipmentItem
from src.http.communication_constants import communication_constants as comm_consts

GUARD_TEMPLATE = ("These guards act as the military and police force of Whiterun, defending Whiterun against "
                  "external and internal threats. They wear yellow cloth over scaled armour and carry shields "
                  "painted with the horse head of Whiterun.")
SHORT_BIO = "Hrolmir lives in Riften. He and several others have been hired by Mercer Frey to guard his house."
RICH_BIO = " ".join(["Lydia is a Nord housecarl sworn to the Thane of Whiterun."] * 10)


def llm_answer(given_name="Hrothgar", bio=None, speech=None) -> str:
    return json.dumps({
        "given_name": given_name,
        "bio": bio or f"{given_name} is a Nord guard in his thirties who walks the walls of Whiterun at night and "
                      f"sends half his pay to his sister in Rorikstead. He distrusts mages and loves sweetrolls.",
        "personality_and_speech": speech or "Dry and sardonic. Says 'by Ysmir' when surprised, calls everyone "
                                            "'friend' with a hint of threat, and clears his throat before bad news.",
    })


class FakeLLM:
    def __init__(self, answers=None, delay=0.0, fail=False):
        self.answers = answers
        self.delay = delay
        self.fail = fail
        self.calls: list[list[dict]] = []
        self.lock = threading.Lock()

    def __call__(self, messages):
        with self.lock:
            self.calls.append(messages)
            n = len(self.calls)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("API down")
        if self.answers:
            return self.answers[(n - 1) % len(self.answers)]
        return llm_answer(given_name=f"Name{n}")


def make_character(name="Whiterun Guard", ref_id="01A2B3", bio="You are a male Nord Whiterun Guard.",
                   is_generic=True, gender=0, race="[Race <NordRace (00013746)>]", voice="MaleGuard", player=False):
    return Character(base_id="0A1B2C", ref_id=ref_id, name=name, gender=gender, race=race, is_player_character=player,
                     bio=bio, is_in_combat=False, is_enemy=False, relationship_rank=0, is_generic_npc=is_generic,
                     ingame_voice_model=voice, tts_voice_model="Male Guard", csv_in_game_voice_model=voice,
                     advanced_voice_model="", voice_accent="en",
                     equipment=Equipment({"body": EquipmentItem("Whiterun Guard's Armor"),
                                          "righthand": EquipmentItem("Steel Sword")}),
                     custom_character_values={})


def template_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "Whiterun Guard", "bio": GUARD_TEMPLATE, "ref_id": None, "base_id": None},
        {"name": "Bandit", "bio": "Bandits are lawbreakers.", "ref_id": None, "base_id": None},
        {"name": "Bandit Outlaw", "bio": "Bandits are lawbreakers.", "ref_id": None, "base_id": None},
        {"name": "Evelyna", "bio": "Evelyna is a vampire.", "ref_id": None, "base_id": None},
        {"name": "Evelyna", "bio": "Evelyna is a vampire.", "ref_id": None, "base_id": None},
        {"name": "Hrolmir", "bio": SHORT_BIO, "ref_id": "0A1111", "base_id": "0A2222"},
        {"name": "Lydia", "bio": RICH_BIO, "ref_id": "0A3333", "base_id": "0A4444"},
    ])


def make_generator(tmp_path, llm, wait_seconds=5.0, scope=BioScope.ALL):
    return BioGenerator(llm_call=llm, model_name="mock-mini", store=BioStore(str(tmp_path / "generated_bios")),
                        template_index=TemplateIndex(template_df()), scope=scope, min_words=40,
                        wait_seconds=wait_seconds)


# ---------- classification ----------

def test_classify_categories():
    index = TemplateIndex(template_df())
    assert classify("You are a male Nord Bandit.", False, index, 40) == BioQuality.MISSING
    assert classify("anything", True, index, 40) == BioQuality.MISSING
    assert classify("", False, index, 40) == BioQuality.MISSING
    assert classify(GUARD_TEMPLATE, False, index, 40) == BioQuality.SHARED          # role name, no ref-id
    assert classify("Bandits are lawbreakers.", False, index, 40) == BioQuality.SHARED  # shared by 2 names
    assert classify("Evelyna is a vampire.", False, index, 40) == BioQuality.WEAK    # same person twice, not a template
    assert classify(SHORT_BIO, False, index, 40) == BioQuality.WEAK
    assert classify(RICH_BIO, False, index, 40) == BioQuality.OK


def test_scope():
    assert BioScope.MISSING_ONLY.covers(BioQuality.MISSING)
    assert not BioScope.MISSING_ONLY.covers(BioQuality.WEAK)
    assert BioScope.MISSING_AND_WEAK.covers(BioQuality.WEAK)
    assert not BioScope.MISSING_AND_WEAK.covers(BioQuality.SHARED)
    assert BioScope.ALL.covers(BioQuality.SHARED)
    assert not BioScope.ALL.covers(BioQuality.OK)
    assert BioScope.from_config("Missing only") == BioScope.MISSING_ONLY
    assert BioScope.from_config("garbage") == BioScope.ALL


# ---------- facts / prompt / parsing ----------

def test_fact_parsing():
    assert parse_race("[Race <DarkElfRace (00013742)>]") == "Dunmer (Dark Elf)"
    assert parse_race("[Race <NordRaceVampire (0008883C)>]") == "Nord"
    assert parse_race("Khajiit") == "Khajiit"
    assert parse_voice_type("[VoiceType <MaleOldGrumpy (00013AD9)>]") == "MaleOldGrumpy"
    assert "old and grumpy" in voice_hint("MaleOldGrumpy")
    assert "child" in voice_hint("FemaleChild")
    assert is_creature_race("[Race <DwarvenSphereRace (00028A2B)>]")
    assert not is_creature_race("[Race <NordRace (00013746)>]")
    assert not is_creature_race("[Race <BretonRaceVampire (0008883A)>]")
    assert hold_for("The Bannered Mare") == "Whiterun Hold"
    assert hold_for("Bleak Falls Barrow") == ""
    assert variety_seed("a") == variety_seed("a")
    seeds = {variety_seed(f"default|Whiterun Guard|{i:06X}")["temperament"] for i in range(30)}
    assert len(seeds) > 5  # guards don't all get the same personality hint


def test_prompt_contains_facts_and_template():
    character = make_character(bio=GUARD_TEMPLATE, is_generic=False)
    facts = facts_from_character(character, "Whiterun", GUARD_TEMPLATE, "w|Whiterun Guard|01A2B3")
    messages = build_messages(facts, BioQuality.SHARED)
    user = messages[1]["content"]
    assert messages[0]["role"] == "system" and "JSON" in messages[0]["content"]
    for expected in ["Whiterun Guard", "Nord", "male", "MaleGuard", "Whiterun Hold", "Steel Sword", "INDIVIDUAL",
                     "military and police force", "given name starting with"]:
        assert expected in user
    named = make_character(name="Hrolmir", bio=SHORT_BIO, is_generic=False)
    user = build_messages(facts_from_character(named, "Riften", SHORT_BIO, "k"), BioQuality.WEAK)[1]["content"]
    assert "given name starting with" not in user and "Mercer Frey" in user and "the Rift" in user


def test_parse_response():
    parsed = parse_response("```json\n" + llm_answer() + "\n```")
    assert parsed.given_name == "Hrothgar" and "Personality & speech:" in parsed.as_text()
    assert parse_response("") is None
    assert parse_response('{"bio": "too short"}') is None
    assert parse_response('{"bio": ') is None  # broken JSON
    assert parse_response(" ".join(["word"] * 400)) is None  # far too long
    plain = parse_response("Hrothgar is a Nord guard of Whiterun who has walked the walls for twenty years and still hates the night shift.")
    assert plain is not None and plain.given_name == ""


# ---------- generator behaviour ----------

def test_missing_bio_triggers_generation_once_and_is_saved(tmp_path):
    llm = FakeLLM()
    gen = make_generator(tmp_path, llm)
    guard = make_character()
    gen.apply([guard], "Prid1", "Whiterun")
    assert len(llm.calls) == 1
    assert guard.bio.startswith("Name1 is a Nord guard")
    assert "Personality & speech:" in guard.bio

    path = gen.store.path_for("Prid1", "Whiterun Guard", "01A2B3")
    entry = json.load(open(path, encoding="utf-8"))
    assert entry["source"] == "generated" and entry["model"] == "mock-mini" and entry["created"]
    assert entry["quality"] == "missing" and entry["mode"] == "replace" and entry["enabled"] is True
    assert entry["race"] == "Nord" and entry["gender"] == "male" and entry["location"] == "Whiterun"

    # same NPC loaded again (new Character object, same ref-id): cache hit, no new call
    again = make_character()
    gen.apply([again], "Prid1", "Whiterun")
    assert len(llm.calls) == 1
    assert again.bio == guard.bio
    # an already processed character passes through untouched
    gen.apply([guard], "Prid1", "Whiterun")
    assert len(llm.calls) == 1


def test_cache_survives_restart_and_user_edits(tmp_path):
    gen = make_generator(tmp_path, FakeLLM())
    gen.apply([make_character()], "Prid1", "Whiterun")
    path = gen.store.path_for("Prid1", "Whiterun Guard", "01A2B3")
    entry = json.load(open(path, encoding="utf-8"))
    entry["bio"] = "Edited by the player: Sven is a tired Nord guard who wants to retire to a farm."
    entry["personality_and_speech"] = ""
    json.dump(entry, open(path, "w", encoding="utf-8"))

    llm2 = FakeLLM()
    gen2 = make_generator(tmp_path, llm2)  # new server process
    guard = make_character()
    gen2.apply([guard], "Prid1", "Whiterun")
    assert llm2.calls == []
    assert guard.bio == entry["bio"]

    entry["enabled"] = False
    json.dump(entry, open(path, "w", encoding="utf-8"))
    guard = make_character()
    gen2.apply([guard], "Prid1", "Whiterun")
    assert llm2.calls == [] and guard.bio == "You are a male Nord Whiterun Guard."


def test_unique_per_ref_id_and_world(tmp_path):
    llm = FakeLLM()
    gen = make_generator(tmp_path, llm)
    a, b = make_character(ref_id="01A2B3"), make_character(ref_id="01A2B4")
    gen.apply([a, b], "Prid1", "Whiterun")
    assert len(llm.calls) == 2
    assert a.bio != b.bio
    # different variety hints for the two guards
    assert llm.calls[0][1]["content"] != llm.calls[1][1]["content"]
    # another playthrough gets its own guard
    c = make_character(ref_id="01A2B3")
    gen.apply([c], "Ulfric2", "Whiterun")
    assert len(llm.calls) == 3
    files = sorted(os.listdir(gen.store.root))
    assert files == ["Prid1", "Ulfric2"]


def test_recycled_ref_id_with_other_actor_regenerates(tmp_path):
    llm = FakeLLM()
    gen = make_generator(tmp_path, llm)
    gen.apply([make_character()], "Prid1", "Whiterun")
    other = make_character(gender=1, race="[Race <ImperialRace (00013744)>]")  # same name+ref, different actor
    gen.apply([other], "Prid1", "Whiterun")
    assert len(llm.calls) == 2


def test_shared_template_is_kept_and_individual_appended(tmp_path):
    llm = FakeLLM()
    gen = make_generator(tmp_path, llm)
    guard = make_character(bio=GUARD_TEMPLATE, is_generic=False)
    gen.apply([guard], "Prid1", "Whiterun")
    assert guard.bio.startswith(GUARD_TEMPLATE)
    assert "About this particular Whiterun Guard: Name1" in guard.bio


def test_weak_bio_is_extended(tmp_path):
    gen = make_generator(tmp_path, FakeLLM())
    npc = make_character(name="Hrolmir", ref_id="0A1111", bio=SHORT_BIO, is_generic=False)
    gen.apply([npc], "Prid1", "Riften")
    assert npc.bio.startswith(SHORT_BIO + "\n")


def test_ok_bio_player_and_scope_are_left_alone(tmp_path):
    llm = FakeLLM()
    gen = make_generator(tmp_path, llm, scope=BioScope.MISSING_ONLY)
    lydia = make_character(name="Lydia", ref_id="0A3333", bio=RICH_BIO, is_generic=False)
    player = make_character(name="Prid", ref_id="000014", bio="", player=True)
    template_guard = make_character(bio=GUARD_TEMPLATE, is_generic=False)
    sphere = make_character(name="Dwemer Sphere", ref_id="0B0000", bio="", race="[Race <DwarvenSphereRace (00028A2B)>]")
    gen.apply([lydia, player, template_guard, sphere], "Prid1", "Whiterun")
    assert llm.calls == []
    assert lydia.bio == RICH_BIO and player.bio == "" and template_guard.bio == GUARD_TEMPLATE


def test_fallback_on_llm_failure(tmp_path):
    llm = FakeLLM(fail=True)
    gen = make_generator(tmp_path, llm)
    guard = make_character()
    gen.apply([guard], "Prid1", "Whiterun")  # must not raise
    assert guard.bio == "You are a male Nord Whiterun Guard."
    assert not os.path.exists(gen.store.path_for("Prid1", "Whiterun Guard", "01A2B3"))
    gen.apply([make_character()], "Prid1", "Whiterun")  # no hammering a failing API in the same session
    assert len(llm.calls) == 1


def test_fallback_on_garbage_answer(tmp_path):
    gen = make_generator(tmp_path, FakeLLM(answers=["Sorry, I can't help with that."]))
    guard = make_character()
    gen.apply([guard], "Prid1", "Whiterun")
    assert guard.bio == "You are a male Nord Whiterun Guard."


def test_slow_llm_does_not_block_and_is_used_later(tmp_path):
    llm = FakeLLM(delay=0.6)
    gen = make_generator(tmp_path, llm, wait_seconds=0.1)
    guard = make_character()
    started = time.time()
    gen.apply([guard], "Prid1", "Whiterun")
    assert time.time() - started < 0.5
    assert guard.bio == "You are a male Nord Whiterun Guard."  # conversation starts with the fallback
    # an update while still generating neither waits again nor starts a second call
    started = time.time()
    gen.apply([make_character()], "Prid1", "Whiterun")
    assert time.time() - started < 0.3
    gen.shutdown(wait_for_pending=True)
    assert len(llm.calls) == 1
    assert guard.bio.startswith("Name1")  # set on the live Character when ready
    nxt = make_character()
    gen2 = make_generator(tmp_path, FakeLLM())
    gen2.apply([nxt], "Prid1", "Whiterun")  # next conversation: cache hit
    assert nxt.bio.startswith("Name1")


def test_wait_zero_is_fully_async(tmp_path):
    llm = FakeLLM(delay=0.2)
    gen = make_generator(tmp_path, llm, wait_seconds=0)
    guard = make_character()
    gen.apply([guard], "Prid1", "Whiterun")
    assert guard.bio == "You are a male Nord Whiterun Guard."
    gen.shutdown(wait_for_pending=True)
    assert os.path.exists(gen.store.path_for("Prid1", "Whiterun Guard", "01A2B3"))


def test_location_from_request():
    request = {comm_consts.KEY_CONTEXT: {comm_consts.KEY_CONTEXT_LOCATION: "Dragonsreach"}}
    assert location_from_request(request, "Skyrim") == "Dragonsreach"
    assert location_from_request({}, "Skyrim") == "Skyrim"


# ---------- wiring ----------

def test_make_llm_call_uses_client_without_lock_and_with_timeout():
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = llm_answer()
    sync_client = MagicMock()
    sync_client.chat.completions.create.return_value = completion
    client = MagicMock()
    client.generate_sync_client.return_value = sync_client
    call = make_llm_call(client, "gpt-4.1-nano")
    assert call([{"role": "user", "content": "hi"}]) == llm_answer()
    kwargs = sync_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4.1-nano" and kwargs["timeout"] > 0
    sync_client.close.assert_called_once()


def test_config_default_off_and_factory(default_config, skyrim):
    assert default_config.bio_generation_enabled is False
    assert default_config.bio_generation_scope == "Missing, weak and shared templates"
    assert create_bio_generator(skyrim, default_config, MagicMock()) is None
    default_config.bio_generation_enabled = True
    client = MagicMock()
    client.model_name = "gpt-4.1-mini"
    generator = create_bio_generator(skyrim, default_config, client)
    assert isinstance(generator, BioGenerator)
    assert generator.store.root.endswith(os.path.join("data", "Skyrim", "generated_bios"))


def test_real_csv_npcs(skyrim, tmp_path):
    """Characters resolved through the real skyrim_characters.csv get the expected treatment."""
    index = TemplateIndex(skyrim.character_df)
    def quality(name, race="[Race <NordRace (00013746)>]", base_id="000000"):
        info = skyrim.load_external_character_info(base_id, name, race, 0, "[VoiceType <MaleGuard (0002F7C3)>]")
        return classify(info.bio, info.is_generic_npc, index, 40)
    assert quality("Whiterun Guard") == BioQuality.SHARED
    assert quality("Bandit") == BioQuality.SHARED
    assert quality("Bandit Outlaw") == BioQuality.SHARED
    assert quality("Necromancer", race="[Race <BretonRace (00013741)>]") == BioQuality.MISSING
    assert quality("Lydia", base_id="0A2C8E") == BioQuality.OK


def test_game_manager_hook_puts_generated_bio_into_system_prompt(default_config, skyrim, llm_client, english_language_info,
                                                                  example_start_conversation_request):
    """End to end through GameStateManager.start_conversation with a mocked bio model (no greeting -> no main LLM call)."""
    from src.game_manager import GameStateManager
    default_config.bio_generation_enabled = True
    default_config.automatic_greeting = False

    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = llm_answer(given_name="Vigdis", bio="Vigdis is a Breton necromancer in her forties "
        "who hides from the Vigil in the barrow, sells soul gems to anyone who pays, and hates spiders more than death.")
    sync_client = MagicMock()
    sync_client.chat.completions.create.return_value = completion
    bio_client = MagicMock()
    bio_client.model_name = "mock-mini"
    bio_client.generate_sync_client.return_value = sync_client

    manager = GameStateManager(skyrim, MagicMock(), default_config, english_language_info, llm_client, summary_client=bio_client)
    request = example_start_conversation_request.model_dump(by_alias=True, exclude_none=True)
    npc = request[comm_consts.KEY_ACTORS][1]
    npc[comm_consts.KEY_ACTOR_NAME] = "Necromancer"            # not in skyrim_characters.csv
    npc[comm_consts.KEY_ACTOR_REFID] = 0x0E1F22
    npc[comm_consts.KEY_ACTOR_RACE] = "[Race <BretonRace (00013741)>]"
    request[comm_consts.KEY_CONTEXT][comm_consts.KEY_CONTEXT_LOCATION] = "Bleak Falls Barrow"
    manager.start_conversation(request)

    talk = manager._GameStateManager__talk
    system_prompt = talk._Conversation__messages._message_thread__messages[0].text
    assert "Vigdis is a Breton necromancer" in system_prompt
    assert sync_client.chat.completions.create.call_count == 1
    user_prompt = sync_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Bleak Falls Barrow" in user_prompt and "Breton" in user_prompt
    saved = os.path.join(os.path.dirname(skyrim.conversation_folder_path), "generated_bios", "Test1", "Necromancer - 0E1F22.json")
    assert json.load(open(saved, encoding="utf-8"))["given_name"] == "Vigdis"
