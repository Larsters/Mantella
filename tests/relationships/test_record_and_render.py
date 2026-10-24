from datetime import datetime

from src.relationships import record as rec
from src.relationships import render
from src.relationships.parser import parse_update_reply


# ---------------- record merge / caps ----------------

def test_new_record_is_neutral():
    r = rec.new_record("Lydia", "A2C94", "Dragonborn")
    assert r["opinion"] == 0 and r["schema_version"] == rec.SCHEMA_VERSION
    assert r["opinion_label"] == "is neutral towards"


def test_apply_update_adds_grudge_with_witnesses_and_day():
    r = rec.new_record("Lydia")
    update = {"opinion_change": -20, "reason": "insulted in public", "attitude": "Cold and curt.",
              "new_grudges": [{"text": "Called me a bitch in front of Alfhild", "severity": 2, "witnesses": ["Alfhild"]}],
              "new_facts": ["Is a member of the Companions"]}
    out = rec.apply_update(r, update, game_day=12.6, now=datetime(2026, 1, 1))
    assert out["opinion"] == -20 and out["opinion_label"] == "is wary of"
    assert out["grudges"][0] == {"id": "g1", "text": "Called me a bitch in front of Alfhild", "day": 12, "severity": 2, "witnesses": ["Alfhild"]}
    assert out["facts"][0]["id"] == "f1"
    assert out["first_met_day"] == 12 and out["last_seen_day"] == 12.6
    assert out["conversation_count"] == 1
    assert out["history"][-1] == {"day": 12, "change": -20, "reason": "insulted in public"}
    assert r["grudges"] == []  # input not mutated


def test_opinion_is_clamped_and_change_limited():
    r = rec.new_record("Guard")
    r["opinion"] = 90
    out = rec.apply_update(r, {"opinion_change": 500}, 1)
    assert out["opinion"] == 100
    out = rec.apply_update(rec.new_record("Guard"), {"opinion_change": -500}, 1)
    assert out["opinion"] == -rec.MAX_OPINION_CHANGE_PER_CONVERSATION
    out = rec.apply_update(rec.new_record("Guard"), {"opinion_change": "garbage"}, 1)
    assert out["opinion"] == 0


def test_duplicates_are_merged_not_appended():
    r = rec.apply_update(rec.new_record("Lydia"), {"new_grudges": [{"text": "Called me a bitch in front of Alfhild", "severity": 1, "witnesses": ["Alfhild"]}]}, 12)
    r = rec.apply_update(r, {"new_grudges": [{"text": "called me a bitch in front of Alfhild!", "severity": 3, "witnesses": ["Mila"]}],
                             "new_facts": ["Likes mead", "likes mead."]}, 14)
    assert len(r["grudges"]) == 1
    assert r["grudges"][0]["severity"] == 3
    assert r["grudges"][0]["witnesses"] == ["Alfhild", "Mila"]
    assert r["grudges"][0]["day"] == 14
    assert len(r["facts"]) == 1


def test_caps_keep_newest_facts_and_most_severe_grudges():
    r = rec.new_record("Lydia")
    for day in range(30):
        r = rec.apply_update(r, {"new_facts": [f"unique fact number {day} about {'x' * day}"],
                                 "new_grudges": [{"text": f"grudge {day} {'y' * day}", "severity": 3 if day == 0 else 1}]}, day)
    assert len(r["facts"]) == rec.CAPS["facts"]
    assert r["facts"][-1]["day"] == 29 and min(f["day"] for f in r["facts"]) == 30 - rec.CAPS["facts"]
    assert len(r["grudges"]) == rec.CAPS["grudges"]
    assert any(g["severity"] == 3 for g in r["grudges"])  # the old but severe grudge survives
    assert len(r["history"]) <= rec.CAPS["history"]


def test_resolve_by_id_and_text():
    r = rec.apply_update(rec.new_record("Lydia"), {
        "new_grudges": [{"text": "Stole my sweetroll"}],
        "new_favours": [{"text": "Lent 100 gold", "owed_by": "player"}],
        "new_promises": [{"text": "Will bring her a dragon bone", "by": "player"}]}, 3)
    r = rec.apply_update(r, {"resolved_grudges": ["g1"], "settled_favours": ["lent 100 gold"],
                             "resolved_promises": [{"id": "p1", "status": "kept"}], "opinion_change": 10}, 5)
    assert r["grudges"] == [] and r["favours"] == [] and r["promises"] == []
    reasons = [h["reason"] for h in r["history"]]
    assert "Forgave: Stole my sweetroll" in reasons
    assert "Promise kept: Will bring her a dragon bone" in reasons


def test_normalize_tolerates_hand_edited_files():
    raw = {"opinion": "250", "facts": ["Is from Morrowind", 42, {"text": ""}], "grudges": [{"text": "Hit me", "severity": 9, "witnesses": "Alfhild, Mila"}],
           "favours": "not a list", "last_seen_day": "yesterday"}
    r = rec.normalize_record(raw, "Lydia", "1")
    assert r["opinion"] == 100
    assert [f["text"] for f in r["facts"]] == ["Is from Morrowind"]  # non-text entries are dropped
    assert r["grudges"][0]["severity"] == 3 and r["grudges"][0]["witnesses"] == ["Alfhild", "Mila"]
    assert r["favours"] == [] and r["last_seen_day"] is None
    assert rec.normalize_record(None, "X")["npc_name"] == "X"


def test_decay_towards_neutral_with_grudge_floor():
    r = rec.new_record("Lydia")
    r.update({"opinion": 50, "last_seen_day": 10})
    assert rec.decayed_opinion(r, 20, 2) == 30
    assert rec.decayed_opinion(r, 20, 0) == 50
    r.update({"opinion": -50, "grudges": [{"id": "g1", "text": "x", "severity": 2}]})
    assert rec.decayed_opinion(r, 100, 5) == -16  # never above -8 * severity while the grudge stands
    r["grudges"] = []
    assert rec.decayed_opinion(r, 100, 5) == 0


# ---------------- rendering ----------------

def _insult_record():
    r = rec.new_record("Lydia", "1", "Dragonborn")
    return rec.apply_update(r, {"opinion_change": -30, "attitude": "Still seething about the insult.",
                                "new_grudges": [{"text": "Dragonborn called me a bitch in front of Alfhild.", "severity": 2, "witnesses": ["Alfhild"]}],
                                "new_facts": ["Is the Thane of Whiterun"],
                                "new_promises": [{"text": "Promised to apologise", "by": "player"}]}, 12.3)


def test_three_days_later_the_insult_is_still_in_the_prompt():
    text = render.render_record(_insult_record(), "Dragonborn", current_day=15.1, token_budget=150)
    assert text.startswith("Lydia dislikes Dragonborn (opinion -30 of 100). Still seething about the insult.")
    assert "Grudge: Dragonborn called me a bitch in front of Alfhild (3 days ago, witnessed by Alfhild)." in text
    assert "Open promise by Dragonborn: Promised to apologise." in text
    assert "Lydia knows about Dragonborn: Is the Thane of Whiterun." in text


def test_render_respects_token_budget_and_priority():
    r = _insult_record()
    for i in range(10):
        r = rec.apply_update(r, {"new_facts": [f"Fact {i}: " + "word " * 15]}, 13)
    text = render.render_record(r, "Dragonborn", 15, token_budget=60)
    assert render.approx_tokens(text) <= 60
    assert "Grudge:" in text  # grudges outrank facts
    tiny = render.render_record(r, "Dragonborn", 15, token_budget=5)
    assert tiny.startswith("Lydia dislikes Dragonborn")  # opinion line is always kept


def test_render_empty_record_is_empty():
    assert render.render_record(rec.new_record("Guard"), "P", 1, 150) == ""
    assert render.render_records([], "P", 1, 150) == ""


def test_render_multiple_records_splits_budget():
    a = _insult_record()
    b = rec.apply_update(rec.new_record("Alfhild"), {"opinion_change": 15, "new_facts": ["Helped with the farm"]}, 12)
    text = render.render_records([a, b], "Dragonborn", 15, token_budget=150)
    assert text.startswith("Relationship with the player:\n")
    assert "Lydia dislikes" in text and "Alfhild likes" in text
    assert render.approx_tokens(text) <= 150 + 10


def test_adjust_trust():
    assert render.adjust_trust("an acquaintance", -40) == "a disliked acquaintance"
    assert render.adjust_trust("a friend", 70) == "a deeply trusted friend"
    assert render.adjust_trust("a stranger", 0) == "a stranger"
    assert render.adjust_trust("an enemy", 90) == "an enemy"
    assert render.adjust_trust("a friend", None) == "a friend"


# ---------------- parser fallback ----------------

def test_parser_handles_messy_replies():
    good = '{"npcs": [{"name": "Lydia", "opinion_change": -5}]}'
    assert parse_update_reply(good)["npcs"][0]["name"] == "Lydia"
    assert parse_update_reply(f"```json\n{good}\n```")["npcs"][0]["opinion_change"] == -5
    assert parse_update_reply(f"Sure! Here's the update:\n{good}\nHope that helps.")["npcs"][0]["name"] == "Lydia"
    assert parse_update_reply('{"npcs": [{"name": "Lydia", "opinion_change": 3,},],}')["npcs"][0]["opinion_change"] == 3
    assert parse_update_reply("{'npcs': [{'name': 'Lydia', 'opinion_change': 2, 'flag': true}]}")["npcs"][0]["opinion_change"] == 2
    assert parse_update_reply('[{"name": "Lydia", "opinion_change": 1}]')["npcs"][0]["opinion_change"] == 1
    assert parse_update_reply('{"name": "Lydia", "opinion_change": 4, "new_facts": []}')["npcs"][0]["opinion_change"] == 4
    assert parse_update_reply('{"npcs": {"Lydia": {"opinion_change": 7}}}')["npcs"][0] == {"name": "Lydia", "opinion_change": 7}
    assert parse_update_reply('Text with a {brace} and then {"npcs": [{"name": "Lydia"}]}')["npcs"] == [{"name": "Lydia"}]
    assert parse_update_reply("I cannot do that.") is None
    assert parse_update_reply("") is None
    assert parse_update_reply(None) is None
