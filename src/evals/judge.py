"""LLM-as-judge: fixed rubric, JSON output, scores 1-5 per criterion. Speed and cost are NOT judged
(they come from replay measurements).

Bias controls (pointwise judging): the judge never sees the model name or the other candidates' replies,
every reply is judged in its own call, jobs are shuffled with a fixed seed, temperature is 0 where the
model supports it, the rubric has anchored levels and says that longer is not better, and one fixed
judge model is used per report.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from src.evals.budget import Budget, Estimate
from src.evals.common import (Pricing, adapt_request_params, append_jsonl, estimate_tokens, message_text, output_cap,
                              read_jsonl, reasoning_allowance)
from src.evals.llm import DEFAULT_BASE_URL, complete_chat

CRITERIA = ('in_character', 'lore_accuracy', 'action_choice', 'voice_fit')
RUBRIC_VERSION = 1
DEFAULT_JUDGE_MODEL = 'gpt-5.4'
DEFAULT_JUDGE_PARAMS = {'temperature': 0, 'max_tokens': 1200, 'reasoning_effort': 'low',
                        'response_format': {'type': 'json_object'}}
EXPECTED_JUDGE_OUTPUT = 220
HISTORY_MESSAGES = 10

JUDGE_SYSTEM = """You are a strict, consistent evaluator of NPC dialogue generated for Mantella, a mod that gives \
voiced, AI-driven conversations to NPCs in The Elder Scrolls V: Skyrim (Fourth Era, year 201). The reply you judge \
is turned into speech by text-to-speech, and tool calls make the NPC act in the game.

Score ONE candidate reply on four criteria, each an integer from 1 to 5:

in_character - Does the reply sound like THIS NPC (bio, personality, speech register, status, relationship to the player, memories)?
  5: unmistakably this person; tone and attitude fit; uses memory naturally. 3: generic but plausible NPC. \
1: breaks character (assistant tone, meta talk, mentions AI/prompts/rules) or contradicts the bio.

lore_accuracy - Is everything said consistent with Skyrim lore, the scene and the bio?
  5: accurate; no anachronisms; no invented world facts; promises only what the game can do. 3: minor stretch or vagueness. \
1: clear lore error, modern concepts treated as real, or a contradiction of the scene or bio.

action_choice - Given the situation and ONLY the tools offered, did the NPC call the right tool(s) with sensible \
arguments, or correctly call none?
  5: exactly the warranted action (or correctly none); arguments valid (names from the Available lists). 3: debatable \
(an unneeded harmless emote, or a needed action missing but the words cover it). 1: a harmful or clearly wrong action \
(attacking over words, following when duty forbids, brawling without the player's consent) or ignoring an explicit request \
that a tool could fulfil.

voice_fit - Is it good spoken dialogue for TTS?
  5: only spoken words; short (1-3 sentences for one NPC; in multi-NPC scenes every line starts with the full speaker name \
like "Lydia: ..."); natural to hear. 3: a bit long or stiff. 1: markdown, lists, asterisks/stage directions, quotation \
marks, emoji, a "Name:" prefix in a one-on-one conversation, or a wall of text.
  If the reply has no spoken text and only tool calls, score voice_fit 3.

Longer is not better. Judge only what is in the reply. If REFERENCE NOTES are given, they describe the ideal behaviour \
for this test case; use them, but reward any reply that meets their intent.

Answer with JSON only, exactly these keys:
{"in_character": 1-5, "lore_accuracy": 1-5, "action_choice": 1-5, "voice_fit": 1-5, "rationale": "<= 60 words"}"""


def _format_tools(tools: list[dict] | None) -> str:
    if not tools:
        return '(no tools were offered in this turn)'
    lines = []
    for tool in tools:
        function = tool.get('function', {})
        params = []
        for name, spec in (function.get('parameters', {}).get('properties') or {}).items():
            description = spec.get('description', '')
            if spec.get('enum'):
                enum = spec['enum']
                description += f" One of {len(enum)} values, e.g. {', '.join(map(str, enum[:6]))}."
            params.append(f'{name}: {description}'.strip())
        lines.append(f"- {function.get('name')}: {function.get('description', '')}" + (f" Params: {'; '.join(params)}" if params else ''))
    return '\n'.join(lines)


def _format_history(messages: list[dict[str, Any]]) -> str:
    rendered = []
    for message in [m for m in messages if m.get('role') != 'system'][-HISTORY_MESSAGES:]:
        role = message.get('role')
        if role == 'tool':
            rendered.append(f'[tool result] {message_text(message)}')
            continue
        text = message_text(message)
        if role == 'assistant' and message.get('tool_calls'):
            calls = ', '.join(f"{c['function']['name']}({c['function'].get('arguments', '')})" for c in message['tool_calls'] if 'function' in c)
            text = (text + ' ' if text else '') + f'[called {calls}]'
        rendered.append(f"{'PLAYER/GAME' if role == 'user' else 'NPC'}: {text}")
    return '\n'.join(rendered) or '(conversation starts here)'


def _format_reply(reply: dict[str, Any]) -> str:
    text = reply.get('content') or ''
    calls = reply.get('tool_calls') or []
    out = f'Spoken text: {text}' if text.strip() else 'Spoken text: (none)'
    if calls:
        out += '\nTool calls:\n' + '\n'.join(f"- {c['function']['name']}({c['function'].get('arguments', '')})" for c in calls)
    else:
        out += '\nTool calls: (none)'
    return out


def build_judge_messages(item: dict[str, Any], reply: dict[str, Any], request_messages: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    messages = request_messages or item['request']['messages']
    system = next((message_text(m) for m in messages if m.get('role') == 'system'), '(none)')
    scenario = item['tags'].get('scenario', 'single')
    parts = [
        f'SCENARIO TYPE: {scenario} ' + {'single': '(player talks with one NPC)', 'multi': '(player talks with several NPCs)',
                                          'radiant': '(NPCs talk with each other; no player)'}.get(scenario, ''),
        f'NPC INSTRUCTIONS AND BACKGROUND (the system prompt the NPC model received):\n<<<\n{system}\n>>>',
        f'TOOLS OFFERED:\n{_format_tools(item["request"].get("tools"))}',
        f'CONVERSATION SO FAR (latest last):\n{_format_history(messages)}',
    ]
    if item.get('expected') and item['expected'].get('notes'):
        parts.append(f"REFERENCE NOTES:\n{item['expected']['notes']}")
    parts.append(f'CANDIDATE REPLY:\n{_format_reply(reply)}')
    parts.append('Return the JSON now.')
    return [{'role': 'system', 'content': JUDGE_SYSTEM}, {'role': 'user', 'content': '\n\n'.join(parts)}]


def parse_judgment(text: str) -> dict[str, Any]:
    """Parses the judge's JSON (tolerates code fences and surrounding text). Raises ValueError."""
    if not text or not text.strip():
        raise ValueError('empty judge output')
    cleaned = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not match:
            raise ValueError('no JSON object in judge output')
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError('judge output is not a JSON object')
    scores: dict[str, int] = {}
    for criterion in CRITERIA:
        value = data.get(criterion)
        if isinstance(value, dict):
            value = value.get('score')
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f'missing or non-numeric score for {criterion}')
        if not 1 <= number <= 5:
            raise ValueError(f'score for {criterion} out of range: {number}')
        scores[criterion] = int(round(number))
    return {'scores': scores, 'rationale': str(data.get('rationale', ''))[:600]}


def latest_records(path: str, key_fields: tuple[str, ...] = ('item_id', 'candidate', 'repeat')) -> dict[tuple, dict[str, Any]]:
    """Later lines win (resumed runs append)."""
    records: dict[tuple, dict[str, Any]] = {}
    for record in read_jsonl(path):
        records[tuple(record.get(k, 0) for k in key_fields)] = record
    return records


def judgeable(replies: dict[tuple, dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in replies.values() if not r.get('skipped') and not r.get('error')
            and ((r.get('content') or '').strip() or r.get('tool_calls'))]


def estimate_judge(items: dict[str, dict[str, Any]], replies: list[dict[str, Any]], judge_model: str,
                   params: dict[str, Any], pricing: Pricing) -> Estimate:
    estimate = Estimate()
    price, known = pricing.lookup(judge_model)
    if not known:
        estimate.unpriced_models.add(judge_model)
    adapted = adapt_request_params(judge_model, params)
    cap = output_cap(adapted, 1200)
    expected_out = min(cap, EXPECTED_JUDGE_OUTPUT + reasoning_allowance(judge_model, adapted))
    for reply in replies:
        item = items.get(reply['item_id'])
        if not item:
            continue
        input_tokens = sum(estimate_tokens(m['content']) for m in build_judge_messages(item, reply))
        estimate.calls += 1
        estimate.input_tokens += input_tokens
        estimate.expected_output_tokens += expected_out
        estimate.max_output_tokens += cap
        estimate.expected_usd += (input_tokens * price['input'] + expected_out * price['output']) / 1e6
        estimate.max_usd += (input_tokens * price['input'] + cap * price['output']) / 1e6
    estimate.lines.append(f'  judge {judge_model:<30} {estimate.calls:>4} calls  ~${estimate.expected_usd:.3f} expected  '
                          f'${estimate.max_usd:.3f} worst case (one retry on bad JSON not included)')
    return estimate


def run_judge(items: dict[str, dict[str, Any]], replies: list[dict[str, Any]], judge_model: str, params: dict[str, Any],
              out_path: str, client_factory: Callable[[str, str | None], Any], pricing: Pricing, budget: Budget,
              base_url: str = DEFAULT_BASE_URL, api_key_env: str | None = None, concurrency: int = 2, seed: int = 7,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    done = {k for k, r in latest_records(out_path, ('item_id', 'candidate', 'repeat', 'judge_model')).items() if r.get('scores')}
    jobs = [r for r in replies if (r['item_id'], r['candidate'], r.get('repeat', 0), judge_model) not in done and r['item_id'] in items]
    random.Random(seed).shuffle(jobs)
    adapted = adapt_request_params(judge_model, params)
    lock = threading.Lock()
    summary = {'judged': 0, 'parse_failures': 0, 'errors': 0, 'already_done': len(done), 'stopped_by_budget': 0}

    def work(reply: dict[str, Any]) -> None:
        record: dict[str, Any] = {'item_id': reply['item_id'], 'candidate': reply['candidate'], 'repeat': reply.get('repeat', 0),
                                  'judge_model': judge_model, 'rubric_version': RUBRIC_VERSION,
                                  'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
        if budget.exhausted:
            with lock:
                summary['stopped_by_budget'] += 1
            return
        client = client_factory(base_url, api_key_env)
        messages = build_judge_messages(items[reply['item_id']], reply)
        usage_total = {'prompt_tokens': 0, 'cached_tokens': 0, 'completion_tokens': 0}
        cost = 0.0
        parsed = None
        error = None
        raw = ''
        for attempt in range(2):  # one retry on unparseable output
            result = complete_chat(client, judge_model, messages, adapted)
            if result['usage']:
                for key in usage_total:
                    usage_total[key] += result['usage'].get(key, 0)
                call_cost = pricing.cost(judge_model, result['usage']['prompt_tokens'], result['usage']['completion_tokens'],
                                         result['usage']['cached_tokens'])
                cost += call_cost
                budget.charge(call_cost)
            if result['error']:
                error = result['error']
                break
            raw = result['content']
            try:
                parsed = parse_judgment(raw)
                error = None
                break
            except ValueError as e:
                error = f'parse error: {e}'
                messages = messages + [{'role': 'assistant', 'content': raw},
                                       {'role': 'user', 'content': 'That was not valid. Reply with only the JSON object with the four integer scores and rationale.'}]
        record.update({'usage': usage_total, 'cost_usd': round(cost, 6)})
        if parsed:
            record.update(parsed)
        else:
            record.update({'error': error, 'raw': raw[:2000]})
        with lock:
            if parsed:
                summary['judged'] += 1
            elif error and error.startswith('parse error'):
                summary['parse_failures'] += 1
            else:
                summary['errors'] += 1
            append_jsonl(out_path, record)
            n = summary['judged'] + summary['parse_failures'] + summary['errors']
            log(f"  [{n}/{len(jobs)}] {reply['candidate']} {reply['item_id']}: "
                + (' '.join(f'{k}={v}' for k, v in parsed['scores'].items()) if parsed else str(error)[:80]))

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(work, jobs))
    summary['spent_usd'] = round(budget.spent, 4)
    return summary
