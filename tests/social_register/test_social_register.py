import json
from types import SimpleNamespace

import pytest
import tiktoken

from src import social_register as sr
from src import speech_flavor as sf
from src.social_register.data import KHAJIIT_TEXT, STREET_TEXT, TAIL
from src.speech_flavor.presets import GERMAN_MIX

NORD = "[Race <NordRace (00013746)>]"
REDGUARD = "[Race <RedguardRace (00013748)>]"
IMPERIAL = "[Race <ImperialRace (00013744)>]"
KHAJIIT = "[Race <KhajiitRace (00013745)>]"
ENC = tiktoken.get_encoding("cl100k_base")


def cfg(enabled=True, flavor="None", intensity="Light", scope="By race", save_folder=""):
    return SimpleNamespace(social_register_enabled=enabled, speech_flavor=flavor, speech_flavor_intensity=intensity,
                           speech_flavor_scope=scope, speech_flavor_tts_respell=True, save_folder=save_folder, game=None)


def npc(name, race=NORD, bio="", voice="MaleEvenToned", base_id="0"):
    return SimpleNamespace(name=name, race=race, bio=bio, in_game_voice_model=voice, base_id=base_id,
                           is_player_character=False)


def compose(config, npcs):
    return sr.compose_speech_flavor(config, npcs, sf.get_speech_flavor_text(config, npcs))


# ------------------------------------------------------------ detection
@pytest.mark.parametrize("bio", [
    "Brennus is a homeless beggar who sleeps by the well.",
    "A member of the Thieves Guild who works the market.",
    "She is a skooma addict living in the Ratway.",
    "He is a dock worker in Windhelm.",
    "Works as a farmhand on Pelagia Farm.",
    "An orphan girl at Honorhall.",
    "A Nord miner at Left Hand Mine.",
    "A moon sugar smuggler.",
])
def test_detect_by_bio_keyword(bio):
    assert sr.detect_register("Somebody", bio, "MaleEvenToned", NORD) == "street"


def test_bio_keyword_only_in_opening():
    bio = "Hrongar is the Jarl's brother. " + "x " * 200 + "He once caught a thief."
    assert sr.detect_register("Hrongar", bio, "MaleEvenToned", NORD) is None
    assert sr.detect_register("Ysolda", "Ysolda is a merchant who wants to buy the Bannered Mare.", "", NORD) is None


@pytest.mark.parametrize("name", ["Brenuin", "Edda", "Snilf", "Angrenor Once-Honored", "Noster Eagle-Eye", "Degaine",
                                  "Silda the Unseen", "Vekel the Man", "Brynjolf", "Delvin Mallory"])
def test_detect_by_known_name(name):
    assert sr.detect_register(name, "", "MaleEvenToned", NORD) == "street"


def test_detect_by_voice_type():
    assert sr.detect_register("Some Drunk", "", "MaleDrunk", REDGUARD) == "street"
    assert sr.detect_register("Townsman", "", "MaleCoward", NORD) is None


@pytest.mark.parametrize("race,voice", [(KHAJIIT, "MaleEvenToned"), ("KhajiitRaceVampire", ""), (None, "FemaleKhajiit")])
def test_khajiit_always_street_khajiit_variant(race, voice):
    assert sr.detect_register("Ri'saad", "Ri'saad leads a trading caravan.", voice, race) == "khajiit"


def test_guards_and_lydia_never_detected():
    assert sr.detect_register("Whiterun Guard", "A thief catcher.", "MaleGuard", NORD) is None
    assert sr.detect_register("Riften Guard", "Takes bribes from the Thieves Guild.", "MaleEvenToned", NORD) is None
    assert sr.detect_register("Somebody", "A former beggar.", "MaleSoldier", NORD) is None
    assert sr.detect_register("Lydia", "Lydia was once a servant.", "FemaleEvenToned", NORD) is None


# ------------------------------------------------------------ rendering / composition
def test_off_by_default():
    assert sr.get_social_register_text(SimpleNamespace(), [npc("Brenuin", bio="beggar")]) == ""
    assert compose(cfg(enabled=False), [npc("Brenuin")]) == ""


def test_single_street_npc_text():
    text = compose(cfg(), [npc("Brenuin", REDGUARD, voice="MaleDrunk")])
    assert text == STREET_TEXT.format(names="Brenuin", s="s") + " " + TAIL
    assert "Flagon" in text and "skooma" in text


def test_ordinary_npc_gets_nothing():
    assert compose(cfg(), [npc("Ysolda", bio="Ysolda is a merchant.")]) == ""


def test_khajiit_text_and_no_german():
    text = compose(cfg(flavor="German mix", intensity="Heavy"), [npc("Ri'saad", KHAJIIT)])
    assert text.startswith("Ri'saad speaks Khajiit street talk")
    assert "this one" in text and "moon sugar" in text
    assert "German" not in text  # W10 by-race scope gives Khajiit no German
    assert text.endswith(TAIL)


def test_german_light_composes_for_nord_beggar():
    config = cfg(flavor="German mix", intensity="Light")
    text = compose(config, [npc("Snilf", NORD)])
    german = GERMAN_MIX.intensities["light"]
    assert text.startswith(german)
    assert text == german + " " + STREET_TEXT.format(names="Snilf", s="s")  # German tail is reused, no double tail
    assert text.count("colons") == 1


def test_multi_npc_names_each_group():
    npcs = [npc("Brenuin", REDGUARD), npc("Edda", NORD), npc("Ri'saad", KHAJIIT), npc("Ysolda", NORD, bio="A merchant.")]
    text = sr.get_social_register_text(cfg(), npcs)
    assert "Brenuin and Edda talk street" in text
    assert "Ri'saad speaks Khajiit street talk" in text
    assert "Ysolda" not in text


def test_player_is_skipped():
    player = SimpleNamespace(name="Player", race=KHAJIIT, bio="", in_game_voice_model="", base_id="7",
                             is_player_character=True)
    assert sr.get_social_register_text(cfg(), [player]) == ""


def test_token_budget_and_parser_safety():
    for template in (STREET_TEXT, KHAJIIT_TEXT):
        full = template.format(names="Angrenor Once-Honored", s="s") + " " + TAIL
        assert len(ENC.encode(full)) <= 90
        for bad in (":", "(", ")", "[", "]", "*", '"', "{", "}"):
            assert bad not in full


def test_compose_never_raises():
    broken = SimpleNamespace(social_register_enabled=True, save_folder="")
    assert sr.compose_speech_flavor(broken, [object()], "keep me") == "keep me"


# ------------------------------------------------------------ per-NPC overrides
@pytest.fixture
def save_folder(tmp_path):
    folder = tmp_path / "data" / "Skyrim" / "character_overrides"
    folder.mkdir(parents=True)
    (folder / "a.json").write_text(json.dumps([
        {"name": "Ysolda", "social_register": "street"},
        {"name": "Brynjolf", "social_register": "none"},
        {"name": "J'zargo", "social_register": "none"},
        {"name": "Whiterun Guard", "speech_flavor": "light"},
    ]), encoding="utf-8")
    (folder / "b.json").write_text(json.dumps({"name": "Edda", "base_id": "0x1234", "social_register": "off"}),
                                   encoding="utf-8")
    sr._override_cache.clear()
    return str(tmp_path)


def test_override_forces_register_even_when_off(save_folder):
    assert compose(cfg(enabled=False, save_folder=save_folder), [npc("Ysolda")]).startswith("Ysolda talks street")
    assert compose(cfg(enabled=False, save_folder=save_folder), [npc("Brenuin")]) == ""


def test_override_disables_register(save_folder):
    config = cfg(save_folder=save_folder)
    assert sr.get_social_register_text(config, [npc("Brynjolf")]) == ""
    assert sr.get_social_register_text(config, [npc("J'zargo", KHAJIIT)]) == ""
    # base_id must match
    assert sr.get_social_register_text(config, [npc("Edda", base_id="1234")]) == ""
    assert sr.get_social_register_text(config, [npc("Edda", base_id="999")]).startswith("Edda talks")


def test_guard_override_without_key_is_ignored(save_folder):
    assert sr.get_social_register_text(cfg(save_folder=save_folder), [npc("Whiterun Guard", voice="MaleGuard")]) == ""


def test_force_street_on_khajiit_keeps_khajiit_variant(tmp_path):
    folder = tmp_path / "data" / "Skyrim" / "character_overrides"
    folder.mkdir(parents=True)
    (folder / "k.json").write_text(json.dumps({"name": "Kharjo", "social_register": "street"}), encoding="utf-8")
    sr._override_cache.clear()
    text = sr.get_social_register_text(cfg(enabled=False, save_folder=str(tmp_path)), [npc("Kharjo", KHAJIIT)])
    assert text.startswith("Kharjo speaks Khajiit street talk")
