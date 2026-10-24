import json
import os
from types import SimpleNamespace

import pytest

from src.character_manager import Character
from src.characters_manager import Characters
from src.llm.message_thread import message_thread
from src.llm.messages import AssistantMessage, UserMessage
from src.llm.sentence import Sentence
from src.llm.sentence_content import SentenceContent, SentenceTypeEnum
from src import relationships
from src.relationships.tracker import RelationshipRememberer, RelationshipStore, RelationshipTracker, build_groups
from src.relationships.updater import RESPONSE_FORMAT


def _npc(name: str, ref_id: str, bio: str = "", is_player: bool = False) -> Character:
    return Character(base_id=ref_id, ref_id=ref_id, name=name, gender=1, race="Nord", is_player_character=is_player,
                     bio=bio, is_in_combat=False, is_enemy=False, relationship_rank=0, is_generic_npc=False,
                     ingame_voice_model="", tts_voice_model="", csv_in_game_voice_model="", advanced_voice_model="",
                     voice_accent="en", equipment=None, custom_character_values={})


def _say(speaker: Character, text: str) -> AssistantMessage:
    msg = AssistantMessage(None)
    msg.add_sentence(Sentence(SentenceContent(speaker, text, SentenceTypeEnum.SPEECH), "", 0.0))
    return msg


class FakeClient:
    """Stands in for ClientBase: records the request params of every call; no network."""

    def __init__(self, replies, fail_structured: bool = False):
        self._request_params = {"max_tokens": 250, "stop": ["#"], "temperature": 0.5}
        self.replies = list(replies)
        self.fail_structured = fail_structured
        self.calls: list[dict] = []

    def _request_call_full(self, messages):
        self.calls.append({"params": dict(self._request_params), "messages": messages.get_openai_messages()})
        if self.fail_structured and "response_format" in self._request_params:
            raise RuntimeError("400 response_format not supported")
        reply = self.replies.pop(0) if self.replies else None
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    def get_count_tokens(self, text):
        return len(text) // 4


INSULT_REPLY = json.dumps({"npcs": [
    {"name": "Lydia", "opinion_change": -25, "reason": "publicly insulted", "attitude": "Furious and humiliated.",
     "new_facts": ["Is the Thane of Whiterun"],
     "new_grudges": [{"text": "Dragonborn called me a bitch in front of Alfhild", "severity": 2, "witnesses": ["Alfhild"]}],
     "resolved_grudges": [], "new_favours": [], "settled_favours": [], "new_promises": [], "resolved_promises": []},
    {"name": "Guard", "opinion_change": -5, "reason": "rude to a housecarl", "attitude": "Keeps an eye on the player.",
     "new_facts": [], "new_grudges": [], "resolved_grudges": [], "new_favours": [], "settled_favours": [],
     "new_promises": [], "resolved_promises": []},
]})


@pytest.fixture
def people():
    return {"player": _npc("Dragonborn", "14", is_player=True), "lydia": _npc("Lydia", "A2C94", "You are Lydia, a housecarl."),
            "guard": _npc("Guard", "0", "You are a guard.")}


def _conversation(people) -> tuple[message_thread, Characters]:
    chars = Characters()
    thread = message_thread(None, "system prompt")
    chars.add_or_update_character(people["player"], len(thread))
    chars.add_or_update_character(people["lydia"], len(thread))
    chars.add_or_update_character(people["guard"], len(thread))
    chars.set_nearby_npcs([{"name": "Alfhild", "distance": 100}])
    thread.add_message(UserMessage(None, "Lydia, you are a useless bitch.", "Dragonborn"))
    thread.add_message(_say(people["lydia"], "How dare you speak to me like that in front of Alfhild!"))
    thread.add_message(UserMessage(None, "I am the Thane of Whiterun, I say what I want.", "Dragonborn"))
    thread.add_message(_say(people["guard"], "Watch your tongue, Thane or not."))
    return thread, chars


def _tracker(tmp_path, client, **kwargs) -> RelationshipTracker:
    return RelationshipTracker(RelationshipStore(str(tmp_path)), client, "Skyrim", "English", run_in_background=False, **kwargs)


def test_store_path_prefers_legacy_folder_only_if_it_exists(tmp_path):
    store = RelationshipStore(str(tmp_path))
    assert store.path_for("Whiterun Guard 2", "1A", "w1").endswith("/w1/Whiterun Guard - 1A/Whiterun Guard_relationship.json")
    os.makedirs(tmp_path / "w1" / "Lydia")
    assert store.path_for("Lydia", "A2C94", "w1").endswith("/w1/Lydia/Lydia_relationship.json")


def test_store_recovers_from_corrupt_file(tmp_path):
    store = RelationshipStore(str(tmp_path))
    path = store.path_for("Lydia", "1", "w")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("{not json")
    record = store.load("Lydia", "1", "w")
    assert record["opinion"] == 0
    assert os.path.exists(path + ".bak")


def test_multi_npc_update_uses_one_structured_call_and_records_witnesses(tmp_path, people):
    thread, chars = _conversation(people)
    client = FakeClient([INSULT_REPLY])
    tracker = _tracker(tmp_path, client)
    rememberer = RelationshipRememberer(SimpleNamespace(save_conversation_state=lambda *a, **k: None,
                                                        get_prompt_text=lambda *a: ""), tracker)
    rememberer.save_conversation_state(thread, [people["lydia"], people["guard"]], chars, "w1", end_timestamp=12.4)

    assert len(client.calls) == 1
    params = client.calls[0]["params"]
    assert params["response_format"] == RESPONSE_FORMAT
    assert "stop" not in params and params["max_tokens"] >= 800
    assert client._request_params == {"max_tokens": 250, "stop": ["#"], "temperature": 0.5}  # shared client untouched
    user_content = client.calls[0]["messages"][1]["content"]
    assert "Characters to update: Lydia, Guard" in user_content
    assert "Alfhild" in user_content  # nearby NPC offered as a witness
    assert "useless bitch" in user_content

    lydia = tracker.store.load("Lydia", "A2C94", "w1")
    guard = tracker.store.load("Guard", "0", "w1")
    assert lydia["opinion"] == -25 and guard["opinion"] == -5
    assert lydia["grudges"][0]["witnesses"] == ["Alfhild", "Guard"]  # other conversation members are witnesses too
    assert lydia["player_name"] == "Dragonborn" and lydia["last_seen_day"] == 12.4

    # Three days later the record is in the prompt and changes the trust wording
    text = relationships.prompt_text(rememberer, [people["lydia"]], "w1", "Dragonborn", 15.2)
    assert "Grudge: Dragonborn called me a bitch in front of Alfhild (3 days ago, witnessed by Alfhild, Guard)." in text
    assert relationships.adjust_trust(rememberer, "an acquaintance", people["lydia"], "w1", 15.2) == "a disliked acquaintance"


def test_falls_back_to_plain_call_when_structured_output_rejected(tmp_path, people):
    thread, chars = _conversation(people)
    client = FakeClient(["Here you go:\n```json\n" + INSULT_REPLY + "\n```"], fail_structured=True)
    tracker = _tracker(tmp_path, client)
    groups = [([people["lydia"], people["guard"]], thread)]
    tracker.update_groups(groups, "w1", "Dragonborn", ["Alfhild"], 3)
    assert len(client.calls) == 2
    assert "response_format" not in client.calls[1]["params"]
    assert tracker.store.load("Lydia", "A2C94", "w1")["opinion"] == -25


def test_unparseable_reply_leaves_records_untouched(tmp_path, people):
    thread, _ = _conversation(people)
    client = FakeClient(["no json here", "still nothing"])
    tracker = _tracker(tmp_path, client)
    tracker.update_groups([([people["lydia"]], thread)], "w1", "Dragonborn", [], 3)
    assert len(client.calls) == 2
    assert not tracker.store.exists("Lydia", "A2C94", "w1")


def test_too_little_dialogue_makes_no_call(tmp_path, people):
    thread = message_thread(None, "sys")
    thread.add_message(_say(people["lydia"], "Hello."))
    client = FakeClient([INSULT_REPLY])
    _tracker(tmp_path, client).update_groups([([people["lydia"]], thread)], "w1", "Dragonborn", [], 3)
    assert client.calls == []


def test_reload_does_not_reprocess_old_messages(tmp_path, people):
    thread, _ = _conversation(people)
    single = json.dumps({"npcs": [{"name": "Lydia", "opinion_change": -10}]})
    client = FakeClient([single, single])
    tracker = _tracker(tmp_path, client)
    tracker.update_groups([([people["lydia"]], thread)], "w1", "Dragonborn", [], 3, is_final=False)  # reload
    thread.add_message(UserMessage(None, "I'm sorry, Lydia.", "Dragonborn"))
    thread.add_message(_say(people["lydia"], "Fine. Just this once."))
    tracker.update_groups([([people["lydia"]], thread)], "w1", "Dragonborn", [], 3, is_final=True)
    second = client.calls[1]["messages"][1]["content"]
    assert "I'm sorry, Lydia." in second and "useless bitch" not in second
    assert tracker.store.load("Lydia", "A2C94", "w1")["conversation_count"] == 2


def test_single_unnamed_entry_is_accepted_for_single_npc(tmp_path, people):
    thread, _ = _conversation(people)
    client = FakeClient([json.dumps({"opinion_change": 12, "new_facts": ["Pays well"]})])
    tracker = _tracker(tmp_path, client)
    tracker.update_groups([([people["lydia"]], thread)], "w1", "Dragonborn", [], 3)
    assert tracker.store.load("Lydia", "A2C94", "w1")["opinion"] == 12


def test_radiant_conversations_are_ignored(tmp_path, people):
    thread, chars = _conversation(people)
    client = FakeClient([INSULT_REPLY])
    inner_calls = []
    rememberer = RelationshipRememberer(SimpleNamespace(save_conversation_state=lambda *a, **k: inner_calls.append(a)), _tracker(tmp_path, client))
    rememberer.save_conversation_state(thread, [people["lydia"]], chars, "w1", is_radiant=True)
    assert len(inner_calls) == 1 and client.calls == []


def test_build_groups_without_summaries_helpers(people):
    thread, chars = _conversation(people)
    groups = build_groups(object(), thread, [people["lydia"], people["guard"], people["player"]], chars)
    assert len(groups) == 1 and [c.name for c in groups[0][0]] == ["Lydia", "Guard"]


def test_create_rememberer_is_off_by_default():
    inner = object()
    assert relationships.create_rememberer(inner, SimpleNamespace(conversation_folder_path="x"), SimpleNamespace(), None, "English") is inner
    config = SimpleNamespace(relationship_tracking_enabled=True, relationship_token_budget=100,
                             relationship_opinion_decay_per_day=0, relationship_affects_trust=True, relationship_update_prompt="")
    wrapped = relationships.create_rememberer(inner, SimpleNamespace(conversation_folder_path="x", game_name_in_filepath="Skyrim"), config, None, "English")
    assert isinstance(wrapped, RelationshipRememberer)
    # Helpers are no-ops for a plain rememberer
    assert relationships.prompt_text(inner, [], "w", "P", 1) == ""
    assert relationships.adjust_trust(inner, "a friend", None, "w", 1) == "a friend"


# ---------------- integration with the real config / Summaries / Context (slower fixtures) ----------------

def test_config_defaults_keep_feature_off(default_config):
    assert default_config.relationship_tracking_enabled is False
    assert default_config.relationship_token_budget == 150
    assert default_config.relationship_affects_trust is True
    assert "{player_name}" in default_config.relationship_update_prompt


def test_npcs_who_heard_different_messages_get_separate_calls(tmp_path, default_config, default_rememberer, people):
    chars = Characters()
    thread = message_thread(default_config, "system prompt")
    chars.add_or_update_character(people["player"], len(thread))
    chars.add_or_update_character(people["guard"], len(thread))
    thread.add_message(UserMessage(default_config, "Morning, guard.", "Dragonborn"))
    thread.add_message(_say(people["guard"], "Citizen."))
    chars.remove_character(people["guard"], len(thread))
    chars.add_or_update_character(people["lydia"], len(thread))
    thread.add_message(UserMessage(default_config, "Lydia, you are a useless bitch.", "Dragonborn"))
    thread.add_message(_say(people["lydia"], "You will regret that."))

    client = FakeClient([json.dumps({"npcs": [{"name": "Guard", "opinion_change": 1}]}),
                         json.dumps({"npcs": [{"name": "Lydia", "opinion_change": -20}]})])
    tracker = _tracker(tmp_path, client)
    groups = build_groups(default_rememberer, thread, [people["guard"], people["lydia"]], chars)
    tracker.update_groups(groups, "w1", "Dragonborn", [], 4)
    assert len(client.calls) == 2
    guard_prompt = client.calls[0]["messages"][1]["content"]
    assert "Characters to update: Guard" in guard_prompt and "useless bitch" not in guard_prompt
    assert tracker.store.load("Lydia", "A2C94", "w1")["opinion"] == -20
    assert tracker.store.load("Guard", "0", "w1")["opinion"] == 1


def test_context_fills_relationship_placeholder_and_trust(tmp_path, default_config, default_rememberer, llm_client, english_language_info, example_characters_pc_to_npc):
    from src.conversation.context import Context
    from src.relationships import record as rec
    tracker = _tracker(tmp_path, FakeClient([]))
    guard = example_characters_pc_to_npc.get_character_by_name("Guard")
    record = rec.apply_update(rec.new_record("Guard", guard.ref_id), {"opinion_change": -30, "new_grudges": [{"text": "Mocked my helmet", "witnesses": ["Alfhild"]}]}, 10)
    tracker.store.save(record, guard.name, guard.ref_id, "1")
    rememberer = RelationshipRememberer(default_rememberer, tracker)
    context = Context("1", default_config, llm_client, rememberer, english_language_info)
    context.add_or_update_characters(example_characters_pc_to_npc.get_all_characters(), message_count=0)

    with_placeholder = context.generate_system_message("Bio. {relationship} Trust: {trust}.", [])
    assert "Guard dislikes Dragonborn (opinion -30 of 100)." in with_placeholder
    assert "Grudge: Mocked my helmet" in with_placeholder
    assert "Trust: a disliked stranger to Guard." in with_placeholder

    without_placeholder = context.generate_system_message("History: {conversation_summary}", [])
    assert "Guard dislikes Dragonborn" in without_placeholder

    # Plain rememberer: placeholder is empty, trust wording is upstream's
    plain = Context("1", default_config, llm_client, default_rememberer, english_language_info)
    plain.add_or_update_characters(example_characters_pc_to_npc.get_all_characters(), message_count=0)
    assert plain.generate_system_message("[{relationship}] {trust}", []) == "[] a stranger to Guard"


def test_real_client_base_receives_response_format_without_blocking(llm_client, people, tmp_path):
    """Goes through the real ClientBase._request_call_full with a mocked OpenAI SDK client."""
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=INSULT_REPLY))])

    fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()), close=lambda: None)
    llm_client.generate_sync_client = lambda: fake_sdk
    llm_client._generation_lock.acquire()  # simulate an NPC reply being streamed right now
    try:
        thread, _ = _conversation(people)
        tracker = _tracker(tmp_path, llm_client)
        tracker.update_groups([([people["lydia"], people["guard"]], thread)], "w1", "Dragonborn", [], 5)
    finally:
        llm_client._generation_lock.release()
    assert captured["response_format"] == RESPONSE_FORMAT
    assert captured["model"] == llm_client.model_name
    assert "stop" not in captured
    assert tracker.store.load("Lydia", "A2C94", "w1")["opinion"] == -25
