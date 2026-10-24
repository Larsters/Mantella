"""Eval toolkit tests. All LLM clients are fakes: no network, no paid calls."""
import json
import os
from types import SimpleNamespace as NS

import pytest

from src.evals import cli, dataset, judge, replay, report, seeds
from src.evals.budget import Budget
from src.evals.common import Pricing, _fallback_adapt, read_jsonl


# ---------------------------------------------------------------- fakes

def _chunk(content=None, tool_calls=None, usage=None, finish=None):
    choices = [NS(delta=NS(content=content, tool_calls=tool_calls), finish_reason=finish)] if (content or tool_calls or finish) else []
    return NS(choices=choices, usage=usage)


class FakeClient:
    """Streams scripted replies for replay calls and returns scripted JSON for judge calls."""

    def __init__(self, judge_outputs=None, reply='Aye, my Thane.', tool=None):
        self.calls = []
        self.judge_outputs = list(judge_outputs or [])
        self.reply = reply
        self.tool = tool
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, stream=False, **params):
        self.calls.append({'model': model, 'messages': messages, 'stream': stream, **params})
        if stream:
            usage = NS(prompt_tokens=1000, completion_tokens=10, prompt_tokens_details=NS(cached_tokens=0), completion_tokens_details=None)
            chunks = [_chunk(self.reply)]
            if self.tool:
                chunks.append(_chunk(tool_calls=[NS(index=0, id='c1', function=NS(name=self.tool, arguments='{"source":["Lydia"]}'))]))
            chunks += [_chunk(finish='stop'), _chunk(usage=usage)]
            return iter(chunks)
        content = self.judge_outputs.pop(0) if self.judge_outputs else json.dumps(
            {'in_character': 4, 'lore_accuracy': 5, 'action_choice': 3, 'voice_fit': 4, 'rationale': 'fine'})
        return NS(choices=[NS(message=NS(content=content))],
                  usage=NS(prompt_tokens=2000, completion_tokens=50, prompt_tokens_details=None, completion_tokens_details=None))


def _transcript_record(conv, idx, user, tools=True, tool_calls=None, error=None, multi=False, has_player=True, image=0, followup=False):
    return {
        'schema': 1, 'ts': f'2026-10-24T12:00:{idx:02d}', 'conversation_id': conv, 'turn_index': idx, 'call_type': 'dialogue',
        'model': 'gpt-4.1-mini', 'speaker': 'Lydia', 'npcs': ['Lydia'], 'has_player': has_player, 'is_multi_npc': multi,
        'request': {'model': 'gpt-4.1-mini', 'messages': [{'role': 'system', 'content': 'You are Lydia.'}, {'role': 'user', 'content': user}],
                    'tools': [{'type': 'function', 'function': {'name': 'Wait', 'description': 'wait'}}] if tools else None,
                    'params': {'max_tokens': 250, 'stop': ['#']}},
        'response': {'content': 'Yes.', 'tool_calls': tool_calls, 'completed': True, 'error': error},
        'timing': {'ttft_s': 0.8, 'total_s': 1.2}, 'usage': {'prompt_tokens': 100, 'completion_tokens': 3, 'cached_tokens': 0},
        'flags': {'images_stripped': image, 'followup_after_tool': followup},
    }


# ---------------------------------------------------------------- dataset / seeds

def test_select_turns_dedupes_tags_and_filters():
    wait_call = [{'id': '1', 'type': 'function', 'function': {'name': 'Wait', 'arguments': '{}'}}]
    records = [
        _transcript_record('c1', 0, 'Wait here.', tool_calls=wait_call),
        _transcript_record('c2', 0, 'wait here.  '),                       # duplicate of the first (case/space)
        _transcript_record('c1', 1, 'Broken', error='APIError'),           # errored
        _transcript_record('c1', 2, 'Look', image=1),                      # had a screenshot
        _transcript_record('c1', 3, 'next', followup=True, tools=False),   # second call
        _transcript_record('c3', 0, 'Hello all', multi=True),
        _transcript_record('c4', 0, 'radiant', multi=True, has_player=False),
    ]
    items, stats = dataset.select_turns(records)
    assert stats['duplicates'] == 1 and stats['errors'] == 1 and stats['images'] == 1 and stats['followups'] == 1
    assert [i['id'] for i in items] == ['c1#0', 'c3#0', 'c4#0']
    assert items[0]['tags']['had_tool_call'] is True and items[0]['tags']['original_tools_called'] == ['Wait']
    assert [i['tags']['scenario'] for i in items] == ['single', 'multi', 'radiant']


def test_stratified_sample_keeps_rare_groups():
    records = [_transcript_record(f'c{i}', 0, f'line {i}') for i in range(20)] + [_transcript_record('r', 0, 'rad', multi=True, has_player=False)]
    items, _ = dataset.select_turns(records, limit=4)
    assert len(items) == 4 and any(i['tags']['scenario'] == 'radiant' for i in items)


def test_seed_items_are_complete_requests():
    items = seeds.build_seed_items()
    assert len(items) == 16 and len({i['id'] for i in items}) == 16
    by_id = {i['id']: i for i in items}
    lydia = by_id['seed-03-lydia-wait']
    assert lydia['request']['messages'][0]['role'] == 'system'
    assert 'Lydia' in lydia['request']['messages'][0]['content']
    tool_names = [t['function']['name'] for t in lydia['request']['tools']]
    assert 'Wait' in tool_names and 'EndConversation' not in tool_names  # 1:1 filter like FunctionManager
    wait = next(t for t in lydia['request']['tools'] if t['function']['name'] == 'Wait')
    assert 'Available: Lydia' in wait['function']['parameters']['properties']['source']['description']
    assert lydia['expected']['expect_tools'] == ['Wait']
    radiant = by_id['seed-07-radiant-guard-adrianne']
    assert radiant['tags']['scenario'] == 'radiant' and 'Eirik' not in radiant['request']['messages'][0]['content']
    multi = by_id['seed-06-market-multi']
    assert multi['request']['messages'][-1]['content'].startswith('Eirik: ')
    assert all('{' not in i['request']['messages'][0]['content'][:50] for i in items)


# ---------------------------------------------------------------- params / pricing

def test_fallback_adapter_handles_reasoning_models():
    adapted = _fallback_adapt('gpt-5.4-mini', {'max_tokens': 250, 'stop': ['#'], 'temperature': 1, 'reasoning_effort': 'low'})
    assert adapted == {'max_completion_tokens': 250, 'reasoning_effort': 'low'}
    assert _fallback_adapt('gpt-4.1-mini', {'max_tokens': 250, 'reasoning_effort': 'low'}) == {'max_tokens': 250}


def test_pricing_prefix_and_unknown_fallback():
    pricing = Pricing()
    price, known = pricing.lookup('gpt-4.1-mini-2025-04-14')
    assert known and price['input'] == 0.40
    _, known = pricing.lookup('some-unknown-model')
    assert not known
    assert pricing.cost('gpt-4.1-mini', 1_000_000, 0) == pytest.approx(0.40)
    assert pricing.cost('gpt-4.1-mini', 1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.10)


# ---------------------------------------------------------------- replay

def test_build_request_overrides():
    item = seeds.build_seed_items()[2]
    candidate = {'label': 'x', 'model': 'gpt-5.4-mini', 'params': {'reasoning_effort': 'minimal', 'stop': None},
                 'system_suffix': 'Keep it short.', 'system_replace': [['Stay in character.', 'Stay in character always.']]}
    request, skip = replay.build_request(item, candidate)
    assert skip is None and request['model'] == 'gpt-5.4-mini'
    try:  # play: W3's real adapter is present (gpt-5.4-mini has no 'minimal' -> 'low', plus reasoning headroom)
        from src.llm import model_capabilities as mc
        assert request['params'] == {'max_completion_tokens': 250 + mc.REASONING_HEADROOM['low'], 'reasoning_effort': 'low'}
    except ImportError:  # feature branch without W3: W9's fallback adapter
        assert request['params'] == {'max_completion_tokens': 250, 'reasoning_effort': 'minimal'}
    assert request['messages'][0]['content'].endswith('Keep it short.')
    assert 'Stay in character always.' in request['messages'][0]['content']
    templated, _ = replay.build_request(item, {'label': 't', 'model': 'gpt-4.1', 'system_prompt_template': 'NPC {name} at {location}. {unknown}'})
    assert templated['messages'][0]['content'] == 'NPC Lydia at Bleak Falls Barrow. '
    _, skip = replay.build_request({**item, 'prompt_vars': None}, {'label': 't', 'model': 'm', 'system_prompt_template': 'x'})
    assert skip


def test_build_request_replaces_stripped_images():
    item = {'id': 'a', 'tags': {}, 'request': {'model': 'm', 'tools': None, 'params': {}, 'messages': [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'look'},
                                     {'type': 'image_url', 'image_url': {'url': '<image omitted: 9 chars base64>'}, 'omitted': True}]}]}}
    request, _ = replay.build_request(item, {'label': 'x', 'model': 'gpt-4.1-mini'})
    assert request['messages'][0]['content'][1] == {'type': 'text', 'text': '[screenshot of the game omitted]'}


def test_run_replay_measures_and_charges(tmp_path):
    items = seeds.build_seed_items()[:3]
    candidates = [{'label': 'a', 'model': 'gpt-4.1-mini'}, {'label': 'b', 'model': 'gpt-4.1'}]
    fake = FakeClient(tool='Wait')
    budget = Budget(1.0)
    out = str(tmp_path / 'replies.jsonl')
    summary = replay.run_replay(items, candidates, 1, out, lambda url, env: fake, Pricing(), budget, concurrency=2, log=lambda m: None)
    assert summary['done'] == 6
    records = list(read_jsonl(out))
    assert all(r['ttft_s'] is not None and r['usage']['prompt_tokens'] == 1000 for r in records)
    assert all(r['content'] == 'Aye, my Thane.' and r['tool_calls'][0]['function']['name'] == 'Wait' for r in records)
    assert all('stream_options' in c and c['tools'] for c in fake.calls)
    lydia = next(r for r in records if r['item_id'] == 'seed-03-lydia-wait')
    assert lydia['action_check']['pass'] is True
    assert budget.spent == pytest.approx(sum(r['cost_usd'] for r in records), rel=1e-3)
    # resume: nothing left to do
    again = replay.run_replay(items, candidates, 1, out, lambda url, env: fake, Pricing(), budget, log=lambda m: None)
    assert again['done'] == 0 and again['already_done'] == 6


def test_run_replay_stops_at_budget(tmp_path):
    items = seeds.build_seed_items()[:4]
    budget = Budget(0.0000001)
    fake = FakeClient()
    summary = replay.run_replay(items, [{'label': 'a', 'model': 'gpt-4.1'}], 1, str(tmp_path / 'r.jsonl'),
                                lambda url, env: fake, Pricing(), budget, concurrency=1, log=lambda m: None)
    assert len(fake.calls) == 1 and summary['stopped_by_budget'] == 3


# ---------------------------------------------------------------- judge

@pytest.mark.parametrize('text,expected', [
    ('{"in_character": 5, "lore_accuracy": 4, "action_choice": 3, "voice_fit": 2, "rationale": "ok"}', [5, 4, 3, 2]),
    ('```json\n{"in_character": "4", "lore_accuracy": 4.0, "action_choice": 5, "voice_fit": 5}\n```', [4, 4, 5, 5]),
    ('Here you go: {"in_character": {"score": 3}, "lore_accuracy": 3, "action_choice": 3, "voice_fit": 3} thanks', [3, 3, 3, 3]),
])
def test_parse_judgment_variants(text, expected):
    assert list(judge.parse_judgment(text)['scores'].values()) == expected


@pytest.mark.parametrize('text', ['', 'no json here', '{"in_character": 9, "lore_accuracy": 4, "action_choice": 3, "voice_fit": 2}',
                                  '{"in_character": 4, "lore_accuracy": 4}', '[1, 2]'])
def test_parse_judgment_rejects_bad_output(text):
    with pytest.raises(ValueError):
        judge.parse_judgment(text)


def test_judge_messages_hide_model_and_include_context():
    item = seeds.build_seed_items()[2]
    reply = {'item_id': item['id'], 'candidate': 'secret-model-label', 'model': 'gpt-secret', 'content': 'As you wish.',
             'tool_calls': [{'function': {'name': 'Wait', 'arguments': '{"source":["Lydia"]}'}}]}
    messages = judge.build_judge_messages(item, reply)
    text = json.dumps(messages)
    assert 'secret' not in text
    assert 'REFERENCE NOTES' in messages[1]['content'] and 'Wait({' in messages[1]['content']
    assert 'Bleak Falls Barrow' in messages[1]['content']


def test_run_judge_retries_once_on_bad_json(tmp_path):
    items = {i['id']: i for i in seeds.build_seed_items()[:2]}
    replies = [{'item_id': k, 'candidate': 'a', 'repeat': 0, 'content': 'Hello.', 'tool_calls': None} for k in items]
    good = json.dumps({'in_character': 5, 'lore_accuracy': 5, 'action_choice': 5, 'voice_fit': 5, 'rationale': 'great'})
    fake = FakeClient(judge_outputs=['not json', good, 'still bad', 'nope'])
    out = str(tmp_path / 'judgments.jsonl')
    summary = judge.run_judge(items, replies, 'gpt-4.1', {'temperature': 0, 'max_tokens': 300}, out, lambda u, e: fake,
                              Pricing(), Budget(1.0), concurrency=1, log=lambda m: None)
    assert summary['judged'] == 1 and summary['parse_failures'] == 1 and len(fake.calls) == 4
    records = list(read_jsonl(out))
    assert sum(1 for r in records if r.get('scores')) == 1
    assert all(r['cost_usd'] > 0 for r in records)


# ---------------------------------------------------------------- report

def test_stats_helpers():
    assert report.mean_ci([4, 4, 4])['ci95'] == 0
    stat = report.mean_ci([3, 5])
    assert stat['mean'] == 4 and stat['ci95'] == pytest.approx(12.706 * 1.41421356 / 1.41421356, rel=1e-3)
    assert report.percentile([1, 2, 3, 4, 5], 50) == 3 and report.percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)
    assert report.mean_ci([])['mean'] is None


def test_report_aggregates_per_candidate(tmp_path):
    items = {i['id']: i for i in seeds.build_seed_items()[:2]}
    ids = list(items)
    replies = []
    judgments = []
    for cand, ttfts, scores in (('fast', [0.5, 0.7], [5, 3]), ('slow', [2.0, 3.0], [2, 2])):
        for item_id, ttft, score in zip(ids, ttfts, scores):
            replies.append({'item_id': item_id, 'candidate': cand, 'repeat': 0, 'model': cand, 'content': 'Hi.', 'tool_calls': None,
                            'ttft_s': ttft, 'total_s': ttft + 1, 'cost_usd': 0.001, 'usage': {'completion_tokens': 10},
                            'voice': {'words': 1, 'tool_only': False}, 'action_check': {'pass': cand == 'fast', 'forbidden_called': []}})
            judgments.append({'item_id': item_id, 'candidate': cand, 'repeat': 0, 'judge_model': 'judge-x',
                              'scores': {c: score for c in judge.CRITERIA}, 'rationale': 'r', 'cost_usd': 0.002})
    replies.append({'item_id': ids[0], 'candidate': 'slow', 'repeat': 1, 'model': 'slow', 'error': 'boom', 'content': ''})
    judgments.append({'item_id': ids[0], 'candidate': 'fast', 'repeat': 0, 'judge_model': 'other-judge', 'scores': {c: 1 for c in judge.CRITERIA}})
    with open(tmp_path / 'replies.jsonl', 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in replies)
    with open(tmp_path / 'judgments.jsonl', 'w') as f:
        f.writelines(json.dumps(j) + '\n' for j in judgments)
    result = report.write_report(str(tmp_path), items)
    assert result['judge_model'] == 'judge-x'  # most used judge
    fast, slow = result['leaderboard']
    assert fast['candidate'] == 'fast' and fast['overall']['mean'] == 4 and fast['criteria']['voice_fit']['n'] == 2
    assert fast['action_rule_pass_rate'] == 1.0 and slow['action_rule_pass_rate'] == 0.0
    assert fast['ttft_p50'] == pytest.approx(0.6) and slow['errors'] == 1
    assert fast['cost_per_reply_usd'] == pytest.approx(0.001)
    html_text = (tmp_path / 'report.html').read_text()
    assert 'prefers-color-scheme' in html_text and 'http' not in html_text.replace('http-equiv', '')
    assert json.loads((tmp_path / 'report.json').read_text())['leaderboard'][0]['candidate'] == 'fast'


# ---------------------------------------------------------------- CLI end to end (fake client)

def test_cli_dry_run_spends_nothing_and_leaves_no_run(tmp_path, capsys):
    data = str(tmp_path)
    assert cli.main(['--data-dir', data, 'build-set', '--name', 'seed', '--seeds']) == 0
    fake = FakeClient()
    code = cli.main(['--data-dir', data, 'run', '--set', 'seed', '--model', 'gpt-4.1-mini', '--max-usd', '5'], client_factory=lambda u, e: fake)
    assert code == 2 and fake.calls == []
    assert 'Dry run' in capsys.readouterr().out
    assert os.listdir(os.path.join(data, 'evals')) == ['sets']


def test_cli_refuses_above_cap(tmp_path, capsys):
    data = str(tmp_path)
    cli.main(['--data-dir', data, 'build-set', '--name', 'seed'])
    fake = FakeClient()
    code = cli.main(['--data-dir', data, 'replay', '--set', 'seed', '--model', 'gpt-5.5', '--max-usd', '0.001', '--yes'],
                    client_factory=lambda u, e: fake)
    assert code == 2 and fake.calls == [] and 'REFUSED' in capsys.readouterr().out


def test_cli_full_run_with_fake_client(tmp_path):
    data = str(tmp_path)
    cli.main(['--data-dir', data, 'build-set', '--name', 'seed'])
    fake = FakeClient()
    code = cli.main(['--data-dir', data, 'run', '--set', 'seed', '--limit', '3', '--model', 'gpt-4.1-mini', '--model', 'gpt-4.1',
                     '--judge-model', 'gpt-4.1', '--max-usd', '1', '--yes'], client_factory=lambda u, e: fake)
    assert code == 0
    [run_id] = [d for d in os.listdir(os.path.join(data, 'evals')) if d != 'sets']
    run_path = os.path.join(data, 'evals', run_id)
    assert len(list(read_jsonl(os.path.join(run_path, 'replies.jsonl')))) == 6
    assert len(list(read_jsonl(os.path.join(run_path, 'judgments.jsonl')))) == 6
    rep = json.loads(open(os.path.join(run_path, 'report.json')).read())
    assert {r['candidate'] for r in rep['leaderboard']} == {'gpt-4.1-mini', 'gpt-4.1'}
    assert cli.main(['--data-dir', data, 'report', '--run', run_id]) == 0
    # judging again finds nothing to do
    assert cli.main(['--data-dir', data, 'judge', '--run', run_id, '--judge-model', 'gpt-4.1', '--yes']) == 0
