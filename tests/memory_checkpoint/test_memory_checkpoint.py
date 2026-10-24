"""Crash-safe memory (src/memory_checkpoint): checkpoint triggers, no double counting with the final pass,
reloads, startup recovery, wiring, and the tightened AddToConversation texts. No network."""
import json
import os
from types import SimpleNamespace

import pytest

from src import memory_checkpoint
from src.characters_manager import Characters
from src.llm.message_thread import message_thread
from src.llm.messages import UserMessage
from src.llm.sentence import Sentence
from src.llm.sentence_content import SentenceContent, SentenceTypeEnum
from src.memory_checkpoint.checkpointer import CheckpointRememberer, JOURNAL_NAME, is_saved_for
from src.relationships.tracker import RelationshipRememberer, RelationshipStore, RelationshipTracker
from src.remember.summaries import Summaries
from tests.relationships.test_tracker import FakeClient, _npc, _say


class FakeInner:
    """Records what each save receives (talk lines only, as the real rememberers read them)."""

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def save_conversation_state(self, messages, npcs_to_summarize, npcs_in_conversation, world_id, is_reload=False,
                                pending_shares=None, end_timestamp=None, is_radiant=False):
        if self.fail:
            raise RuntimeError("summary model down")
        self.calls.append({"talk": [m.get_formatted_content() for m in messages.get_talk_only()],
                           "npcs": [n.name for n in npcs_to_summarize], "is_reload": is_reload, "thread": messages,
                           "shares": pending_shares, "log": npcs_in_conversation.get_participation_log(), "world": world_id})

    def get_prompt_text(self, characters, world_id):
        return ""


class FakeTimer:
    created: list["FakeTimer"] = []

    def __init__(self, seconds, fn, args=()):
        self.seconds, self.fn, self.args, self.cancelled, self.daemon = seconds, fn, args, False, False
        FakeTimer.created.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.fn(*self.args)


@pytest.fixture
def people():
    return {"player": _npc("Eirik", "14", is_player=True), "guard": _npc("Whiterun Guard", "1A", "You are a guard."),
            "lydia": _npc("Lydia", "A2C94", "You are Lydia.")}


class Talk:
    """A minimal live conversation: thread + Characters + the context the hook receives."""

    def __init__(self, people, rememberer, npcs=("guard",)):
        self.people = people
        self.rememberer = rememberer
        self.chars = Characters()
        self.thread = message_thread(None, "system prompt")
        self.chars.add_or_update_character(people["player"], len(self.thread))
        for key in npcs:
            self.chars.add_or_update_character(people[key], len(self.thread))
        self.context = SimpleNamespace(config=SimpleNamespace(conversation_summary_enabled=True), npcs_in_conversation=self.chars,
                                       world_id="w1", game_days=12.5)
        self.turn = 0

    def player(self, text=None):
        self.turn += 1
        self.thread.add_message(UserMessage(None, text or f"player line {self.turn}", "Eirik"))
        self.rememberer.on_player_turn(self.thread, self.context)

    def npc(self, text=None, who="guard"):
        self.thread.add_message(_say(self.people[who], text or f"npc line {self.turn}"))

    def end(self, shares=None):
        self.rememberer.save_conversation_state(self.thread, self.chars.get_non_player_characters(), self.chars, "w1",
                                                False, shares, 12.6, False)


def _wrapper(tmp_path, inner, every=3, idle=0.0, **kwargs):
    FakeTimer.created = []
    return CheckpointRememberer(inner, str(tmp_path / "memory_checkpoints"), every_n_turns=every, idle_minutes=idle,
                                run_in_background=False, timer_factory=FakeTimer, **kwargs)


# ---------- triggers ----------
def test_checkpoint_after_n_turns_covers_only_complete_lines_before_the_new_player_line(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=3))
    for _ in range(2):
        talk.player(); talk.npc()
    talk.player()  # 3rd turn: only 4 earlier lines, below the summary minimum -> waits
    assert inner.calls == []
    talk.npc(); talk.player()  # 4th turn: 6 complete lines
    assert len(inner.calls) == 1
    call = inner.calls[0]
    assert call["talk"] == ["player line 1", "npc line 1", "player line 2", "npc line 2", "player line 3", "npc line 3"]
    assert call["npcs"] == ["Whiterun Guard"] and call["is_reload"] is False and call["world"] == "w1"
    assert len(call["thread"]) == len(talk.thread) - 1  # cut before the new player line; indices still fit
    assert not is_saved_for(talk.thread[-1], "Whiterun Guard")  # the newest line is left for later


def test_turn_counter_restarts_after_a_checkpoint(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=3))
    for _ in range(4):
        talk.player(); talk.npc()
    assert len(inner.calls) == 1
    for _ in range(2):
        talk.player(); talk.npc()
    assert len(inner.calls) == 1  # only 2 turns since
    talk.player(); talk.npc(); talk.player()
    assert len(inner.calls) == 2
    assert inner.calls[1]["talk"][0] == "player line 4"  # starts right after the first checkpoint


def test_idle_checkpoint_saves_everything_including_the_last_player_line(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=0, idle=3))
    talk.player(); talk.npc(); talk.player(); talk.npc()
    assert len(FakeTimer.created) == 2 and FakeTimer.created[0].cancelled and FakeTimer.created[1].seconds == 180
    talk.player("Is anyone there?")  # then the game crashes: no reply, no end
    FakeTimer.created[-1].fire()
    assert len(inner.calls) == 1 and inner.calls[0]["talk"][-1] == "Is anyone there?"
    # The conversation ends normally later: nothing left, no second save
    talk.end()
    assert len(inner.calls) == 1


def test_ended_conversation_cancels_the_idle_timer(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=0, idle=3))
    for _ in range(3):
        talk.player(); talk.npc()
    talk.end()
    assert FakeTimer.created[-1].cancelled
    assert len(inner.calls) == 1  # the normal end-of-conversation save only


def test_summaries_disabled_means_no_checkpoints(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=1))
    talk.context.config.conversation_summary_enabled = False
    for _ in range(5):
        talk.player(); talk.npc()
    assert inner.calls == []


# ---------- no double counting ----------
def test_final_pass_only_gets_lines_after_the_last_checkpoint(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=3))
    for _ in range(4):
        talk.player(); talk.npc()
    talk.end()
    assert len(inner.calls) == 2
    final = inner.calls[1]
    assert final["talk"] == ["player line 4", "npc line 4"]
    assert final["is_reload"] is True  # lower threshold: a short rest after a checkpoint is still summarised
    first = set(inner.calls[0]["talk"])
    assert not first & set(final["talk"])


def test_without_checkpoints_the_final_save_is_passed_through_unchanged(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=10))
    talk.player(); talk.npc()
    talk.end(shares=[("Whiterun Guard", "Lydia", "A2C94")])
    assert len(inner.calls) == 1
    assert inner.calls[0]["thread"] is talk.thread and inner.calls[0]["is_reload"] is False
    assert inner.calls[0]["shares"] == [("Whiterun Guard", "Lydia", "A2C94")]


def test_reply_still_streaming_when_marked_is_saved_again_in_full(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=0, idle=3))
    for _ in range(3):
        talk.player(); talk.npc()
    FakeTimer.created[-1].fire()  # marks "npc line 3"
    talk.thread[-1].add_sentence(Sentence(SentenceContent(people["guard"], "And one more thing.", SentenceTypeEnum.SPEECH), "", 0.0))
    talk.end()
    assert inner.calls[1]["talk"] == ["npc line 3 And one more thing."]


def test_reload_and_its_deep_copied_tail_are_not_counted_again(tmp_path, people):
    inner = FakeInner()
    rememberer = _wrapper(tmp_path, inner, every=3)
    talk = Talk(people, rememberer)
    for _ in range(4):
        talk.player(); talk.npc()  # checkpoint at turn 4 (lines 1-3)
    # Context window full: Conversation saves with is_reload=True, then keeps a deep-copied tail
    rememberer.save_conversation_state(talk.thread, talk.chars.get_non_player_characters(), talk.chars, "w1", True, None, 12.6, False)
    assert inner.calls[1]["talk"] == ["player line 4", "npc line 4"] and inner.calls[1]["is_reload"] is True
    talk.thread.reload_message_thread("new prompt", lambda msgs, pct: len(msgs) > 4, 0.1)
    talk.player("After the reload."); talk.npc("Indeed.")
    talk.end()
    assert inner.calls[2]["talk"] == ["After the reload.", "Indeed."]


def test_departed_npc_save_marks_lines_only_for_that_npc(tmp_path, people):
    inner = FakeInner()
    rememberer = _wrapper(tmp_path, inner, every=50)
    talk = Talk(people, rememberer, npcs=("guard", "lydia"))
    talk.player("Both of you, listen."); talk.npc("Aye.")
    talk.chars.remove_character(people["guard"], len(talk.thread))
    rememberer.save_conversation_state(talk.thread, [people["guard"]], talk.chars, "w1", True, None, 12.6, False)
    talk.player("Lydia, now just us."); talk.npc("Yes, my Thane.", who="lydia")
    talk.end()
    final = inner.calls[-1]
    assert final["npcs"] == ["Lydia"]
    assert "Both of you, listen." in final["talk"] and "Lydia, now just us." in final["talk"]


def test_failed_checkpoint_leaves_lines_for_the_final_pass(tmp_path, people):
    inner = FakeInner(fail=True)
    talk = Talk(people, _wrapper(tmp_path, inner, every=3))
    for _ in range(4):
        talk.player(); talk.npc()
    inner.fail = False
    talk.end()
    assert len(inner.calls) == 1 and inner.calls[0]["talk"][0] == "player line 1"


def test_relationship_update_counts_each_line_once(tmp_path, people):
    """Checkpoint + final pass through the real W6 wrapper: two update calls, disjoint transcripts."""
    reply = json.dumps({"npcs": [{"name": "Whiterun Guard", "opinion_change": -5, "reason": "rude"}]})
    client = FakeClient([reply, reply])
    tracker = RelationshipTracker(RelationshipStore(str(tmp_path / "conversations")), client, "Skyrim", "English", run_in_background=False)
    summaries = FakeInner()
    inner = RelationshipRememberer(summaries, tracker)
    wrapper = memory_checkpoint.create_rememberer(inner, SimpleNamespace(conversation_folder_path=str(tmp_path / "conversations")),
                                                  SimpleNamespace(memory_checkpoint_enabled=True, memory_checkpoint_every_n_turns=3,
                                                                  memory_checkpoint_idle_minutes=0, memory_checkpoint_recover_on_startup=False))
    assert isinstance(wrapper, CheckpointRememberer) and tracker.dedupe_reloads is False
    wrapper._background = False
    talk = Talk(people, wrapper)
    talk.player("You smell like a troll."); talk.npc("Watch it.")
    talk.player("Yes."); talk.npc("Move along.")
    talk.player("Make me."); talk.npc("Last warning.")
    talk.player("Yes.")  # turn 4: checkpoint of the first six lines
    talk.npc("Then go.")
    talk.end()
    assert len(client.calls) == 2
    first, second = (c["messages"][1]["content"] for c in client.calls)
    assert "troll" in first and "troll" not in second
    assert "Yes." in second and "Then go." in second  # a repeated "Yes." is not dropped as already seen
    record = tracker.store.load("Whiterun Guard", "1A", "w1")
    assert record["opinion"] == -10 and len(summaries.calls) == 2


def test_real_summaries_append_checkpoint_and_rest_to_the_same_file(tmp_path, default_config, people):
    class SummaryClient:
        token_limit = 128000

        def __init__(self):
            self.inputs = []

        def request_call(self, messages):
            self.inputs.append(messages[-1].text)
            return f"Summary {len(self.inputs)}."

        def get_count_tokens(self, text):
            return len(text) // 4

    client = SummaryClient()
    game = SimpleNamespace(conversation_folder_path=str(tmp_path / "conversations"), game_name_in_filepath="Skyrim")
    summaries = Summaries(game, default_config, client, "English")
    talk = Talk(people, _wrapper(tmp_path, summaries, every=3))
    for _ in range(4):
        talk.player(); talk.npc()
    talk.end()
    assert len(client.inputs) == 2
    assert "player line 1" in client.inputs[0] and "player line 4" not in client.inputs[0]
    assert "player line 4" in client.inputs[1] and "player line 1" not in client.inputs[1]
    with open(tmp_path / "conversations" / "w1" / "Whiterun Guard - 1A" / "Whiterun Guard_summary_1.txt", encoding="utf-8") as f:
        text = f.read()
    assert text.index("Summary 1.") < text.index("Summary 2.")


# ---------- journal and startup recovery ----------
def test_journal_holds_unsaved_lines_and_is_removed_at_the_end(tmp_path, people):
    inner = FakeInner()
    talk = Talk(people, _wrapper(tmp_path, inner, every=3))
    path = tmp_path / "memory_checkpoints" / JOURNAL_NAME
    talk.player("Hello there."); talk.npc("Greetings."); talk.player("Seen any dragons?")
    journal = json.loads(path.read_text(encoding="utf-8"))
    assert journal["world_id"] == "w1" and journal["npcs"][0] == {"name": "Whiterun Guard", "ref_id": "1A", "bio": "You are a guard."}
    assert [p["content"] for p in journal["pending"]] == ["Eirik: Hello there.", "Whiterun Guard: Greetings.", "Eirik: Seen any dragons?"]
    talk.npc(); talk.player(); talk.npc(); talk.player()  # checkpoint (6 lines): the journal keeps only the newest line
    journal = json.loads(path.read_text(encoding="utf-8"))
    assert [p["content"] for p in journal["pending"]] == ["Eirik: player line 4"] and journal["checkpoints"] == 1
    talk.end()
    assert not path.exists()


def test_startup_recovery_saves_a_crashed_servers_conversation_once(tmp_path, people):
    talk = Talk(people, _wrapper(tmp_path, FakeInner(), every=50))
    talk.player("I stole the Jarl's crown."); talk.npc("You what?"); talk.player("Just kidding.")
    path = tmp_path / "memory_checkpoints" / JOURNAL_NAME
    journal = json.loads(path.read_text(encoding="utf-8"))
    # Same process: this is the live conversation, not a crashed one
    assert _wrapper(tmp_path, FakeInner()).recover_unfinished() is False and path.exists()

    journal["pid"] = -1  # written by a server process that died
    path.write_text(json.dumps(journal), encoding="utf-8")
    inner = FakeInner()
    assert _wrapper(tmp_path, inner).recover_unfinished() is True
    assert len(inner.calls) == 1
    call = inner.calls[0]
    assert call["npcs"] == ["Whiterun Guard"] and call["world"] == "w1" and call["is_reload"] is True
    assert call["talk"] == ["Eirik: I stole the Jarl's crown.", "Whiterun Guard: You what?", "Eirik: Just kidding."]
    assert not path.exists() and not os.path.exists(str(path) + ".recovering")
    assert _wrapper(tmp_path, inner).recover_unfinished() is False and len(inner.calls) == 1


def test_recovery_feeds_real_summaries_and_relationships(tmp_path, default_config, people):
    folder = tmp_path / "memory_checkpoints"
    folder.mkdir()
    (folder / JOURNAL_NAME).write_text(json.dumps({
        "pid": -1, "conversation_id": "x", "world_id": "w1", "player_name": "Eirik", "game_days": 4.5,
        "npcs": [{"name": "Lydia", "ref_id": "A2C94", "bio": "You are Lydia."}], "nearby": [],
        "pending": [{"role": "user", "content": "Eirik: I promise to bring you a sword."},
                    {"role": "assistant", "content": "Lydia: I will hold you to that."}]}), encoding="utf-8")

    class SummaryClient:
        token_limit = 128000
        inputs = []

        def request_call(self, messages):
            self.inputs.append(messages[-1].text)
            return "Eirik promised Lydia a sword."

        def get_count_tokens(self, text):
            return 10

    game = SimpleNamespace(conversation_folder_path=str(tmp_path / "conversations"), game_name_in_filepath="Skyrim")
    reply = json.dumps({"npcs": [{"name": "Lydia", "opinion_change": 5, "new_promises": [{"text": "Bring Lydia a sword", "by": "player"}]}]})
    tracker = RelationshipTracker(RelationshipStore(game.conversation_folder_path), FakeClient([reply]), "Skyrim", "English", run_in_background=False)
    rememberer = RelationshipRememberer(Summaries(game, default_config, SummaryClient(), "English"), tracker)
    wrapper = memory_checkpoint.create_rememberer(rememberer, game, SimpleNamespace(memory_checkpoint_enabled=True), recover_in_background=False)
    assert isinstance(wrapper, CheckpointRememberer)
    assert "I promise to bring you a sword" in SummaryClient.inputs[0]
    summary = (tmp_path / "conversations" / "w1" / "Lydia - A2C94" / "Lydia_summary_1.txt").read_text(encoding="utf-8")
    assert "promised Lydia a sword" in summary
    assert tracker.store.load("Lydia", "A2C94", "w1")["promises"][0]["text"] == "Bring Lydia a sword"


# ---------- wiring and config ----------
def test_off_by_default_and_hook_is_a_no_op_for_plain_rememberers(people):
    inner = FakeInner()
    assert memory_checkpoint.create_rememberer(inner, SimpleNamespace(conversation_folder_path="x"), SimpleNamespace()) is inner
    memory_checkpoint.on_player_turn(inner, message_thread(None, "x"), None)  # must not raise


def test_wrapper_forwards_unknown_attributes(tmp_path):
    inner = SimpleNamespace(save_conversation_state=None, get_prompt_text=lambda c, w: "memories", relationship_tracker="T")
    wrapper = _wrapper(tmp_path, inner)
    assert wrapper.relationship_tracker == "T" and wrapper.get_prompt_text([], "w") == "memories"


def test_journal_folder_is_next_to_conversations():
    game = SimpleNamespace(conversation_folder_path="/x/data/Skyrim/conversations")
    assert memory_checkpoint.journal_folder(game) == "/x/data/Skyrim/memory_checkpoints"


def test_config_defaults(default_config):
    assert default_config.memory_checkpoint_enabled is False
    assert default_config.memory_checkpoint_every_n_turns == 6
    assert default_config.memory_checkpoint_idle_minutes == 3
    assert default_config.memory_checkpoint_recover_on_startup is True


def test_conversation_calls_the_turn_hook():
    import inspect
    from src.conversation import conversation
    source = inspect.getsource(conversation)
    assert "memory_checkpoint.on_player_turn(self.__rememberer, self.__messages, self.__context)" in source


# ---------- tighter AddToConversation ----------
def test_add_to_conversation_texts_forbid_pulling_in_bystanders():
    from pathlib import Path
    from src import trust
    from src.trust import rules
    action = json.loads((Path(__file__).resolve().parents[2] / "data" / "actions" / "add_to_conversation.json").read_text(encoding="utf-8"))
    for text in (action["guided_description"], action["trust_description"], rules.FUNCTION_BLOCK):
        assert "by name or role" in text
        assert "guard" in text
        assert "nearby" in text and "mentioned" in text
    assert "explicitly call" in action["guided_description"] and "explicitly called" in action["trust_description"]
    assert "Mentioning someone is not asking for them" in rules.FUNCTION_BLOCK
    assert len(action["guided_description"].split()) <= 60
    assert trust.tool_description(action, SimpleNamespace(trust_and_honesty_enabled=True)) == action["trust_description"]
    assert trust.tool_description(action, SimpleNamespace(guided_action_descriptions=False)) == "Add nearby NPC(s) to the conversation."
