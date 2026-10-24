import json
from types import SimpleNamespace

import pytest

from src import speech_flavor as sf
from src.speech_flavor.presets import GERMAN_MIX, find_preset, preset_labels


def cfg(flavor="German mix", intensity="Medium", scope="By race", respell=True, save_folder=""):
    return SimpleNamespace(speech_flavor=flavor, speech_flavor_intensity=intensity, speech_flavor_scope=scope,
                           speech_flavor_tts_respell=respell, save_folder=save_folder, game=None)


def npc(name, race, base_id="0"):
    return SimpleNamespace(name=name, race=race, base_id=base_id, is_player_character=False)


NORD = "[Race <NordRace (00013746)>]"
IMPERIAL = "[Race <ImperialRace (00013744)>]"
KHAJIIT = "[Race <KhajiitRace (00013745)>]"


# ------------------------------------------------------------ presets / parsing
def test_preset_lookup_and_labels():
    assert preset_labels()[0] == "None"
    assert "German mix" in preset_labels()
    assert find_preset("german_mix") is GERMAN_MIX
    assert find_preset("German mix") is GERMAN_MIX
    assert find_preset("None") is None and find_preset("") is None


@pytest.mark.parametrize("race,expected", [
    (NORD, "nord"), ("NordRaceVampire", "nord"), ("Dark Elf", "dunmer"), ("[Race <HighElfRace (x)>]", "altmer"),
    (KHAJIIT, "khajiit"), ("ArgonianRace", "argonian"), ("ElderRace", "imperial"), ("", None), ("DremoraRace", None),
])
def test_normalize_race(race, expected):
    assert sf.normalize_race(race) == expected


def test_parse_override():
    assert sf.parse_override("none") == (True, None, None)
    assert sf.parse_override("heavy") == (False, None, "heavy")
    assert sf.parse_override("german_mix:light") == (False, GERMAN_MIX, "light")
    assert sf.parse_override("German mix") == (False, GERMAN_MIX, None)


# ------------------------------------------------------------ rendering
def test_disabled_by_default_returns_empty():
    assert sf.get_speech_flavor_text(cfg(flavor="None"), [npc("Ulfric", NORD)]) == ""
    assert sf.get_speech_flavor_text(SimpleNamespace(), [npc("Ulfric", NORD)]) == ""


@pytest.mark.parametrize("intensity", ["Light", "Medium", "Heavy"])
def test_each_intensity_renders_its_own_text(intensity):
    text = sf.get_speech_flavor_text(cfg(intensity=intensity), [npc("Ulfric", NORD)])
    assert text == GERMAN_MIX.intensities[intensity.lower()]
    # parser-safety and playability rules are always present
    assert "colons" in text and "actions normally" in text and "clear in English" in text


def test_intensity_texts_differ_and_are_short():
    texts = list(GERMAN_MIX.intensities.values())
    assert len(set(texts)) == 3
    for t in texts:
        assert len(t.split()) <= 60  # ~80 tokens
        assert ":" in t  # only the 'Speech style:' label
        assert t.count(":") == 1


def test_scope_by_race():
    c = cfg(intensity="Medium", scope="By race")
    assert sf.get_speech_flavor_text(c, [npc("Ulfric", NORD)]) == GERMAN_MIX.intensities["medium"]
    assert sf.get_speech_flavor_text(c, [npc("Guard", IMPERIAL)]) == GERMAN_MIX.intensities["light"]
    assert sf.get_speech_flavor_text(c, [npc("J'zargo", KHAJIIT)]) == ""
    # lighter never drops below light
    assert sf.get_speech_flavor_text(cfg(intensity="Light"), [npc("Guard", IMPERIAL)]) == GERMAN_MIX.intensities["light"]
    # unknown race -> configured intensity
    assert sf.get_speech_flavor_text(c, [npc("Dremora", "DremoraRace")]) == GERMAN_MIX.intensities["medium"]


def test_scope_all_npcs_ignores_race():
    c = cfg(intensity="Heavy", scope="All NPCs")
    for race in (NORD, IMPERIAL, KHAJIIT):
        assert sf.get_speech_flavor_text(c, [npc("X", race)]) == GERMAN_MIX.intensities["heavy"]


def test_multi_npc_mixed_levels():
    text = sf.get_speech_flavor_text(cfg(intensity="Heavy"), [
        npc("Ulfric", NORD), npc("Galmar", NORD), npc("Guard", IMPERIAL), npc("J'zargo", KHAJIIT)])
    assert text.startswith("Only Ulfric and Galmar speak like this.")
    assert GERMAN_MIX.intensities["heavy"] in text
    assert "Guard use only about one German word per reply." in text
    assert "J'zargo speak normally, without German." in text


def test_multi_npc_same_level_has_no_name_lines():
    text = sf.get_speech_flavor_text(cfg(), [npc("Ulfric", NORD), npc("Galmar", NORD)])
    assert text == GERMAN_MIX.intensities["medium"]


def test_player_is_ignored():
    player = SimpleNamespace(name="Dragonborn", race=KHAJIIT, base_id="7", is_player_character=True)
    assert sf.get_speech_flavor_text(cfg(), [player, npc("Ulfric", NORD)]) == GERMAN_MIX.intensities["medium"]


# ------------------------------------------------------------ per-NPC overrides
@pytest.fixture
def save_folder(tmp_path):
    folder = tmp_path / "data" / "Skyrim" / "character_overrides"
    folder.mkdir(parents=True)
    (folder / "lydia.json").write_text(json.dumps({"name": "Lydia", "speech_flavor": "heavy"}), encoding="utf-8")
    (folder / "list.json").write_text(json.dumps([
        {"name": "J'zargo", "speech_flavor": "german_mix:light"},
        {"name": "Ulfric Stormcloak", "base_id": "0001414D", "speech_flavor": "none"},
        {"name": "Nazeem", "bio": "no flavour key here"},
    ]), encoding="utf-8")
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    return str(tmp_path)


def test_override_intensity_beats_race_and_global(save_folder):
    c = cfg(intensity="Light", save_folder=save_folder)
    assert sf.get_speech_flavor_text(c, [npc("Lydia", NORD)]) == GERMAN_MIX.intensities["heavy"]
    assert sf.get_speech_flavor_text(c, [npc("J'zargo", KHAJIIT)]) == GERMAN_MIX.intensities["light"]


def test_override_none_and_base_id_matching(save_folder):
    c = cfg(save_folder=save_folder)
    assert sf.get_speech_flavor_text(c, [npc("Ulfric Stormcloak", NORD, base_id="1414D")]) == ""
    # other base id -> override does not apply
    assert sf.get_speech_flavor_text(c, [npc("Ulfric Stormcloak", NORD, base_id="999")]) == GERMAN_MIX.intensities["medium"]


def test_override_works_while_global_flavor_off(save_folder):
    c = cfg(flavor="None", save_folder=save_folder)
    assert sf.get_speech_flavor_text(c, [npc("Lydia", NORD)]) == GERMAN_MIX.intensities["heavy"]
    assert sf.get_speech_flavor_text(c, [npc("Nazeem", "Redguard")]) == ""


# ------------------------------------------------------------ TTS respelling
def test_respell_for_tts():
    out = sf.respell_for_tts("Ja, natürlich, mein Freund! Scheiße. Genau. Danke schön. Großartig.", GERMAN_MIX)
    assert out == "Yah, nah-toor-lish, mine froynd! Shy-sa. Geh-now. Dahnkeh shern. Grossartig."


def test_respell_keeps_names_and_english_words():
    text = "Ja'darri and Jarl Balgruuf was right. The Jagged Crown."
    assert sf.respell_for_tts(text, GERMAN_MIX) == text


def test_prepare_for_tts_is_gated():
    line = "Ja, mein Freund."
    assert sf.prepare_for_tts(line, cfg(flavor="None")) == line
    assert sf.prepare_for_tts(line, cfg(respell=False)) == line
    assert sf.prepare_for_tts(line, SimpleNamespace()) == line
    assert sf.prepare_for_tts(line, cfg()) == "Yah, mine froynd."


# ------------------------------------------------------------ German text vs. output parsers
def test_german_reply_passes_sentence_parsers():
    from src.llm.output.clean_sentence_parser import clean_sentence_parser
    from src.llm.output.sentence_end_parser import sentence_end_parser
    from src.llm.output.output_parser import sentence_generation_settings

    reply = "Ja, natürlich, mein Freund! Was ist los? Scheiße, I become a sweetroll and it is wunderbar. "
    speaker = npc("Ulfric", NORD)
    settings = sentence_generation_settings(speaker)
    _, cleaned = clean_sentence_parser().cut_sentence(reply, settings)
    assert cleaned == reply
    parser = sentence_end_parser()
    sentences, rest = [], cleaned
    while True:
        content, rest = parser.cut_sentence(rest, settings)
        if content is None:
            break
        sentences.append(content.text.strip())
    assert sentences == ["Ja, natürlich, mein Freund!", "Was ist los?",
                         "Scheiße, I become a sweetroll and it is wunderbar."]
