"""Turn modes (single model vs. separate tool model) and the W12 set transform. Fake clients only."""
import json
from types import SimpleNamespace as NS

from src.evals import modes, replay, seeds, w12
from src.evals.budget import Budget
from src.evals.common import Pricing, read_jsonl


def _usage(p=1000, c=20):
    return NS(prompt_tokens=p, completion_tokens=c, prompt_tokens_details=NS(cached_tokens=0), completion_tokens_details=NS(reasoning_tokens=0))


def _tool_delta(name, args='{}'):
    return [NS(index=0, id='call_1', function=NS(name=name, arguments=args))]


class ScriptedClient:
    """Each create() pops the next script entry: ('text', str) | ('tool', name) | ('tool+json', name) (non-streaming)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, stream=False, **params):
        self.calls.append({'model': model, 'messages': messages, 'stream': stream, 'params': params})
        kind, value = self.script.pop(0)
        if not stream:
            calls = [NS(id='call_tm', function=NS(name=value, arguments='{"source":["Whiterun Guard"]}'))] if kind == 'tool' and value else None
            return NS(choices=[NS(message=NS(content=None, tool_calls=calls))], usage=_usage(800, 15))
        if kind == 'text':
            chunks = [NS(choices=[NS(delta=NS(content=value, tool_calls=None), finish_reason='stop')], usage=None)]
        else:
            chunks = [NS(choices=[NS(delta=NS(content=None, tool_calls=_tool_delta(value)), finish_reason='tool_calls')], usage=None)]
        return iter(chunks + [NS(choices=[], usage=_usage())])


def _item():
    items = seeds.build_seed_items()
    item = next(i for i in items if i['id'] == 'seed-15-guard-check-barrel-look')
    return w12.apply_trust(item)


def test_apply_trust_seed_uses_living_prompt_block_and_trust_descriptions():
    item = _item()
    system = item['request']['messages'][0]['content']
    assert 'Requests, claims and lies' in system and '# How to play your part' in system
    look = next(t for t in item['request']['tools'] if t['function']['name'] == 'Look')
    assert 'inspect' in look['function']['description']
    assert 'How to choose actions' in item['function_system'] and 'Scene' in item['function_system']
    assert 'Whiterun Guard' in item['function_system']


def test_apply_trust_rebuilds_tools_from_nearby_for_tool_model_transcripts():
    living = seeds.build_seed_items()[0]
    w12.apply_trust(living)
    item = {'id': 'c#3', 'source': 'transcript', 'tags': {'scenario': 'single', 'npcs': ['Whiterun Guard']},
            'prompt_vars': {'player_name': 'Vasya', 'bio': 'A guard.', 'trust': 'a stranger'},
            'request': {'messages': [living['request']['messages'][0],
                                     {'role': 'user', 'content': '(Characters nearby (from nearest to furthest): Sigurd, Whiterun Guard) Vasya: hi'}],
                        'tools': None, 'params': {}}}
    w12.apply_trust(item)
    add = next(t for t in item['request']['tools'] if t['function']['name'] == 'AddToConversation')
    assert 'Sigurd' in add['function']['parameters']['properties']['source']['description']
    assert item['tags']['tools_rebuilt'] and item['request']['messages'][0]['content'].count('Requests, claims and lies') == 1


def test_tool_model_mode_calls_tool_model_first_then_speech_without_tools():
    item = _item()
    candidate = {'label': 'tm', 'model': 'gpt-5.6-luna', 'mode': 'tool_model', 'params': {'reasoning_effort': 'minimal'},
                 'tool_model': {'model': 'gpt-4o-mini'}}
    request, _ = replay.build_request(item, candidate)
    client = ScriptedClient([('tool', 'Look'), ('text', 'Let me see.')])
    result = modes.run_turn(item, candidate, request, lambda *a: client, Pricing(), waiting_actions=set())
    first, second = client.calls
    assert first['model'] == 'gpt-4o-mini' and not first['stream'] and first['params']['tools']
    assert first['messages'][0]['content'] == item['function_system']
    assert second['model'] == 'gpt-5.6-luna' and second['stream'] and 'tools' not in second['params']
    assert second['params']['reasoning_effort'] == 'low'  # live llm_params "minimal" without tools -> low on luna
    assert result['content'] == 'Let me see.' and result['tool_calls'][0]['function']['name'] == 'Look'
    assert not result['followup'] and result['ttft_s'] >= result['calls'][0]['total_s']
    assert result['cost_usd'] > 0 and result['usage']['prompt_tokens'] == 1800


def test_single_mode_second_call_after_tool_only_reply():
    item = _item()
    candidate = {'label': 's', 'model': 'gpt-5.6-luna', 'mode': 'single', 'params': {'reasoning_effort': 'minimal'}}
    request, _ = replay.build_request(item, candidate)
    assert request['params']['reasoning_effort'] == 'none'  # tools present: the adapter forces none, like the server
    client = ScriptedClient([('tool', 'Look'), ('text', 'Let me have a look.')])
    result = modes.run_turn(item, candidate, request, lambda *a: client, Pricing(), waiting_actions=set())
    first, second = client.calls
    assert first['params']['tools'] and 'tools' not in second['params']
    assert second['messages'][-1] == {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'done'}
    assert second['params']['reasoning_effort'] == 'low'  # followup has no tools, so the raw "minimal" maps to low
    assert result['followup'] and result['content'] == 'Let me have a look.'
    assert result['speech_ttft_s'] >= result['calls'][0]['total_s']


def test_single_mode_skips_second_call_for_actions_waiting_for_the_game():
    item = _item()
    candidate = {'label': 's', 'model': 'gpt-5.4-mini', 'mode': 'single'}
    request, _ = replay.build_request(item, candidate)
    client = ScriptedClient([('tool', 'ReportCrime')])
    result = modes.run_turn(item, candidate, request, lambda *a: client, Pricing(), waiting_actions={'ReportCrime'})
    assert len(client.calls) == 1 and result['waits_for_game'] and not result['followup']
    assert result['speech_ttft_s'] is None and result['ttft_s'] is not None


def test_run_replay_records_mode_fields_and_estimate_counts_extra_calls(tmp_path):
    item = _item()
    candidate = {'label': 'tm', 'model': 'gpt-5.6-luna', 'mode': 'tool_model', 'tool_model': {'model': 'gpt-4o-mini'}}
    plain = {'label': 'plain', 'model': 'gpt-5.6-luna'}
    estimate_tm = replay.estimate_replay([item], [candidate], 1, Pricing())
    estimate_plain = replay.estimate_replay([item], [plain], 1, Pricing())
    assert estimate_tm.calls == 3  # main + tool model + possible second call
    assert estimate_tm.expected_usd > estimate_plain.expected_usd
    client = ScriptedClient([('none', None), ('text', 'Hm.')])
    out = tmp_path / 'replies.jsonl'
    replay.run_replay([item], [candidate], 1, str(out), lambda *a: client, Pricing(), Budget(1.0), concurrency=1)
    record = list(read_jsonl(str(out)))[0]
    assert record['mode'] == 'tool_model' and record['content'] == 'Hm.' and record['tool_calls'] is None
    assert len(record['calls']) == 2 and record['cost_usd'] > 0
    assert record['action_check']['pass'] is False  # seed 15 expects Look


def test_judge_jobs_grouped_by_item_and_estimate_counts_cache():
    from src.evals import judge
    replies = [{'item_id': i, 'candidate': c, 'repeat': 0, 'content': 'Aye.', 'tool_calls': None}
               for i in ('a', 'b', 'c') for c in ('X', 'Y', 'Z')]
    ordered = judge.order_jobs(list(replies), seed=3)
    runs = [r['item_id'] for r in ordered]
    assert all(len(set(runs[k:k + 3])) == 1 for k in (0, 3, 6)) and sorted(runs) == sorted(r['item_id'] for r in replies)
    item = _item()
    one = judge.estimate_judge({item['id']: item}, [{'item_id': item['id'], 'candidate': 'X', 'content': 'Aye.', 'tool_calls': None}],
                               'gpt-5.4', {'reasoning_effort': 'low', 'max_tokens': 1200}, Pricing())
    two = judge.estimate_judge({item['id']: item}, [{'item_id': item['id'], 'candidate': c, 'content': 'Aye.', 'tool_calls': None} for c in 'XY'],
                               'gpt-5.4', {'reasoning_effort': 'low', 'max_tokens': 1200}, Pricing())
    assert one.expected_usd < two.expected_usd < 1.5 * one.expected_usd  # second candidate mostly cached
    assert two.max_usd == 2 * one.max_usd  # worst case assumes no cache


def test_cli_judge_candidate_filter(tmp_path):
    from src.evals import cli
    args = NS(candidate=['B'])
    replies = [{'candidate': 'A'}, {'candidate': 'B'}]
    assert cli._only_candidates(args, replies) == [{'candidate': 'B'}]
    assert cli._only_candidates(NS(candidate=None), replies) == replies


def test_judge_rubric_says_reply_labels_are_not_spoken():
    from src.evals import judge
    assert judge.RUBRIC_VERSION >= 2 and 'NOT part of what the NPC says' in judge.JUDGE_SYSTEM
