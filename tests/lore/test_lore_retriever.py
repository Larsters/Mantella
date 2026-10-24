"""Pure retrieval tests for src/lore (no config, no LLM, no network)."""
import json
import os
import re

import pytest

from src.lore.corpus import LoreNote, load_corpus, VALID_SCOPES
from src.lore.retriever import LoreQuery, LoreRetriever, estimate_tokens, format_note, normalize_race

LORE_DIR = os.path.join("data", "Skyrim", "lore")


@pytest.fixture(scope="module")
def corpus() -> dict[str, LoreNote]:
    return load_corpus([LORE_DIR])


@pytest.fixture(scope="module")
def retriever(corpus) -> LoreRetriever:
    return LoreRetriever(corpus)


def all_ids(retriever: LoreRetriever, query: LoreQuery, max_notes=3, max_tokens=200) -> set[str]:
    scene = retriever.scene_notes(query, max_notes, max_tokens)
    topic = retriever.topic_notes(query, max_notes, max_tokens, exclude={n.id for n in scene})
    return {n.id for n in scene} | {n.id for n in topic}


# --------------------------------------------------------------------------- corpus quality
def test_corpus_size_and_schema(corpus):
    assert 150 <= len(corpus) <= 260
    for note in corpus.values():
        assert note.scope in VALID_SCOPES
        assert note.aliases, note.id
        assert note.source.startswith("https://en.uesp.net/wiki/"), note.id
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", note.text) if s]
        assert 1 <= len(sentences) <= 4, (note.id, len(sentences))
        assert estimate_tokens(format_note(note)) <= 90, note.id


def test_corpus_references_resolve(corpus):
    for note in corpus.values():
        if note.parent:
            assert note.parent in corpus, (note.id, note.parent)
        for place in note.places:
            assert place in corpus and corpus[place].scope == "place", (note.id, place)


def test_corpus_files_carry_attribution():
    for name in os.listdir(LORE_DIR):
        if name.endswith(".json"):
            with open(os.path.join(LORE_DIR, name), encoding="utf-8") as f:
                assert "CC BY-SA" in json.load(f)["attribution"]


def test_required_topics_present(corpus):
    required = ["whiterun_hold", "haafingar", "the_reach", "eastmarch", "the_rift", "the_pale", "winterhold_hold",
                "hjaalmarch", "falkreath_hold", "companions", "college_of_winterhold", "thieves_guild",
                "dark_brotherhood", "stormcloaks", "imperial_legion", "thalmor", "dawnguard", "volkihar", "greybeards",
                "blades", "silver_blood_family", "forsworn", "battle_born", "gray_mane", "east_empire_company",
                "great_war", "white_gold_concordat", "markarth_incident", "death_of_torygg", "talos_ban", "race_dunmer"]
    assert not [r for r in required if r not in corpus]


def test_spoilers_marked_and_hidden_by_default(corpus, retriever):
    spoilers = {nid for nid, n in corpus.items() if n.spoiler}
    assert {"paarthurnax", "companions_werewolves", "alduin_returned"} <= spoilers
    assert not spoilers & set(retriever.notes)
    assert spoilers <= set(LoreRetriever(corpus, allow_spoilers=True).notes)


# --------------------------------------------------------------------------- scenarios
def test_whiterun_asking_about_the_war(retriever):
    q = LoreQuery(location="Whiterun", utterance="What do you think about the war?")
    ids = all_ids(retriever, q)
    assert {"whiterun", "civil_war", "balgruuf"} <= ids


def test_markarth_brings_forsworn_and_silver_blood(retriever):
    q = LoreQuery(location="Markarth")
    scene = [n.id for n in retriever.scene_notes(q, 3, 200)]
    assert scene[0] == "markarth"
    assert {"forsworn", "silver_blood_family"} <= set(scene)


def test_sublocation_maps_to_city(retriever):
    assert retriever.location_chain("Understone Keep") == ["markarth", "the_reach"]
    chain = retriever.location_chain("Dragonsreach")
    assert chain[:2] == ["whiterun", "whiterun_hold"]


def test_longer_location_alias_wins(retriever):
    assert retriever.location_chain("Whiterun Hold") == ["whiterun_hold"]


def test_asking_about_talos(retriever):
    q = LoreQuery(location="Riverwood", utterance="Is it safe to pray to Talos here?")
    topic = [n.id for n in retriever.topic_notes(q, 3, 200)]
    assert topic[0] == "talos_ban"
    q2 = LoreQuery(location="Riverwood", utterance="Will the Thalmor arrest me for worshipping Talos?")
    assert {"talos_ban", "thalmor"} <= {n.id for n in retriever.topic_notes(q2, 3, 200)}


def test_small_talk_injects_nothing(retriever):
    for line in ["Hello there.", "Goodbye.", "How are you today?", "Yes.", ""]:
        assert retriever.topic_notes(LoreQuery(location="Whiterun", utterance=line)) == [], line


def test_budget_and_note_limit(retriever):
    q = LoreQuery(location="Windhelm", utterance="Tell me about Ulfric, the Stormcloaks, Talos and the Thalmor")
    for max_notes, budget in [(3, 200), (1, 200), (3, 80), (3, 10)]:
        notes = retriever.topic_notes(q, max_notes, budget)
        assert len(notes) <= max_notes
        assert sum(estimate_tokens(format_note(n)) for n in notes) <= budget
    assert retriever.topic_notes(q, 3, 10) == []


def test_exclude_and_self_notes(retriever):
    q = LoreQuery(location="Whiterun", utterance="What do you think about the war?", npc_names=["Balgruuf the Greater"])
    ids = {n.id for n in retriever.topic_notes(q)}
    assert "balgruuf" not in ids  # the NPC's own bio already covers them
    ids2 = {n.id for n in retriever.topic_notes(LoreQuery(location="Whiterun", utterance="What about the war?"), exclude={"civil_war"})}
    assert "civil_war" not in ids2


def test_race_notes_from_game_race_strings(retriever):
    assert normalize_race("<Race DarkElfRace (00013742)>") == "Dark Elf"
    q = LoreQuery(location="Windhelm", player_race="<Race ArgonianRace (00013740)>")
    assert "race_argonian" in {n.id for n in retriever.scene_notes(q, 3, 250)}
    # Nords are the default in Skyrim; no note for them
    assert "race_nord" not in retriever.score_scene(LoreQuery(npc_races=["NordRace"]))


def test_bio_faction_head(retriever):
    q = LoreQuery(location="Dawnstar", npc_bios=["Aela is a member of the Companions, a huntress of Jorrvaskr."])
    assert "companions" in {n.id for n in retriever.scene_notes(q)}


def test_previous_npc_line_contributes(retriever):
    q = LoreQuery(location="Riften", utterance="Tell me more.", previous_line="The Dark Brotherhood? Hush, don't speak of them.")
    assert "dark_brotherhood" in {n.id for n in retriever.topic_notes(q)}


def test_personal_override_folder(tmp_path, corpus):
    (tmp_path / "mine.json").write_text(json.dumps({"notes": [
        {"id": "whiterun", "title": "Whiterun", "scope": "place", "aliases": ["Whiterun"], "text": "My own Whiterun."},
        {"id": "nazeem", "disabled": True},
        {"id": "broken"},
    ]}), encoding="utf-8")
    merged = load_corpus([LORE_DIR, str(tmp_path)])
    assert merged["whiterun"].text == "My own Whiterun."
    assert "nazeem" not in merged and "broken" not in merged
    assert len(merged) == len(corpus) - 1
