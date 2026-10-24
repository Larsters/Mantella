"""Replay eval-set turns against candidate models/settings and record replies with timing, tokens and cost."""
from __future__ import annotations

import copy
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from src.evals.budget import Budget, Estimate
from src.evals import modes
from src.evals.checks import action_check, voice_metrics
from src.evals.common import (Pricing, adapt_request_params, append_jsonl, estimate_request_tokens, output_cap,
                              read_jsonl, reasoning_allowance)
from src.evals.llm import DEFAULT_BASE_URL, stream_chat

EXPECTED_REPLY_TOKENS = 120  # typical spoken reply incl. tool-call JSON; used only for the expected-cost estimate
SERVER_ONLY_PARAMS = ('stream', 'stream_options', 'tools', 'n')


class _KeepMissing(dict):
    def __missing__(self, key: str) -> str:
        return ''


def load_candidates(path: str | None, models: list[str] | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if path:
        with open(os.path.expanduser(path), encoding='utf-8') as f:
            data = json.load(f)
        candidates += data['candidates'] if isinstance(data, dict) else data
    for model in models or []:
        candidates.append({'label': model, 'model': model})
    labels = set()
    for candidate in candidates:
        candidate.setdefault('label', candidate['model'])
        if candidate['label'] in labels:
            raise ValueError(f"Duplicate candidate label '{candidate['label']}'")
        labels.add(candidate['label'])
    if not candidates:
        raise ValueError('No candidates: pass --model (repeatable) and/or --candidates file.json')
    return candidates


def _template_text(value: str) -> str:
    if value.startswith('@'):
        with open(os.path.expanduser(value[1:]), encoding='utf-8') as f:
            return f.read()
    return value


def build_request(item: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Returns (request, skip_reason). The request is what gets sent: model, messages, tools, adapted params."""
    recorded = item['request']
    messages = copy.deepcopy(recorded['messages'])
    for message in messages:  # stripped screenshots cannot be replayed: keep a text marker instead
        if isinstance(message.get('content'), list):
            parts = []
            for part in message['content']:
                if isinstance(part, dict) and part.get('type') == 'image_url' and (part.get('omitted') or str(part.get('image_url', {}).get('url', '')).startswith('<image omitted')):
                    parts.append({'type': 'text', 'text': '[screenshot of the game omitted]'})
                elif isinstance(part, dict):
                    part = {k: v for k, v in part.items() if k != 'cache_control'}
                    parts.append(part)
            message['content'] = parts
    system_index = next((i for i, m in enumerate(messages) if m.get('role') == 'system'), None)
    if candidate.get('system_prompt_template'):
        if not item.get('prompt_vars'):
            return None, 'system_prompt_template needs prompt_vars (seed items only)'
        if system_index is None:
            return None, 'no system message to replace'
        template = _template_text(candidate['system_prompt_template'])
        messages[system_index]['content'] = template.format_map(_KeepMissing(item['prompt_vars']))
    if system_index is not None and isinstance(messages[system_index].get('content'), str):
        for old, new in candidate.get('system_replace') or []:
            messages[system_index]['content'] = messages[system_index]['content'].replace(old, new)
        if candidate.get('system_suffix'):
            messages[system_index]['content'] += '\n' + candidate['system_suffix']
    params = {k: v for k, v in (recorded.get('params') or {}).items() if k not in SERVER_ONLY_PARAMS}
    for key, value in (candidate.get('params') or {}).items():
        if value is None:
            params.pop(key, None)
        else:
            params[key] = value
    model = candidate['model']
    tools = recorded.get('tools') if candidate.get('tools', True) else None
    raw_params = dict(params)
    if tools and candidate.get('mode') != 'tool_model':  # the server adapts with tools present (e.g. forces effort none)
        params = adapt_request_params(model, {**params, 'tools': tools})
        params.pop('tools', None)
    else:
        params = adapt_request_params(model, params)
    return {'model': model, 'messages': messages, 'tools': tools or None, 'params': params, 'raw_params': raw_params}, None


def estimate_replay(items: list[dict[str, Any]], candidates: list[dict[str, Any]], repeats: int,
                    pricing: Pricing) -> Estimate:
    total = Estimate()
    for candidate in candidates:
        part = Estimate()
        for item in items:
            request, skip = build_request(item, candidate)
            if skip:
                continue
            input_tokens = estimate_request_tokens(request['messages'], request['tools'])
            cap = output_cap(request['params'], 250)
            allowance = reasoning_allowance(candidate['model'], request['params'])
            expected_out = min(cap, EXPECTED_REPLY_TOKENS + allowance)
            price, known = pricing.lookup(candidate['model'])
            if candidate.get('price'):
                price, known = candidate['price'], True
            if not known:
                part.unpriced_models.add(candidate['model'])
            extra = modes.estimate_extra_calls(candidate, input_tokens, estimate_request_tokens([], request['tools']),
                                               ((input_tokens * price['input'] + expected_out * price['output']) / 1e6,
                                                (input_tokens * price['input'] + cap * price['output']) / 1e6), pricing)
            for _ in range(repeats):
                part.calls += 1 + extra[0]
                part.expected_usd += extra[1]
                part.max_usd += extra[2]
                part.input_tokens += input_tokens
                part.expected_output_tokens += expected_out
                part.max_output_tokens += cap
                part.expected_usd += (input_tokens * price['input'] + expected_out * price['output']) / 1e6
                part.max_usd += (input_tokens * price['input'] + cap * price['output']) / 1e6
        part.lines.append(f"  replay {candidate['label']:<28} {part.calls:>4} calls  ~${part.expected_usd:.3f} expected  "
                          f"${part.max_usd:.3f} worst case")
        total.add(part)
    return total


def done_keys(path: str) -> set[tuple[str, str, int]]:
    return {(r['item_id'], r['candidate'], r.get('repeat', 0)) for r in read_jsonl(path) if not r.get('error') and not r.get('skipped')}


def run_replay(items: list[dict[str, Any]], candidates: list[dict[str, Any]], repeats: int, out_path: str,
               client_factory: Callable[[str, str | None], Any], pricing: Pricing, budget: Budget,
               concurrency: int = 2, seed: int = 7, log: Callable[[str], None] = print) -> dict[str, Any]:
    finished = done_keys(out_path)
    jobs = [(item, candidate, r) for r in range(repeats) for item in items for candidate in candidates
            if (item['id'], candidate['label'], r) not in finished]
    random.Random(seed).shuffle(jobs)  # interleave candidates so time-of-day/API load doesn't favour one model
    lock = threading.Lock()
    waiting_actions = modes.actions_waiting_for_game()
    summary = {'done': 0, 'errors': 0, 'skipped': 0, 'already_done': len(finished), 'stopped_by_budget': 0}

    def work(job: tuple[dict[str, Any], dict[str, Any], int]) -> None:
        item, candidate, repeat = job
        record: dict[str, Any] = {'item_id': item['id'], 'candidate': candidate['label'], 'repeat': repeat,
                                  'model': candidate['model'], 'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
        request, skip = build_request(item, candidate)
        if skip:
            record['skipped'] = skip
        elif budget.exhausted:
            record['skipped'] = 'budget exhausted'
        if record.get('skipped'):
            with lock:
                summary['stopped_by_budget' if record['skipped'] == 'budget exhausted' else 'skipped'] += 1
                append_jsonl(out_path, record)
            return
        if candidate.get('mode'):  # server turn modes: tool model first, or a second call after tool-only replies
            result = modes.run_turn(item, candidate, request, client_factory, pricing, waiting_actions)
            cost = result['cost_usd']
            budget.charge(cost)
            usage = {}
        else:
            client = client_factory(candidate.get('base_url', DEFAULT_BASE_URL), candidate.get('api_key_env'))
            result = stream_chat(client, request['model'], request['messages'], request['tools'], request['params'])
            usage = result['usage'] or {}
            cost = None
        if usage:
            if candidate.get('price'):
                p = candidate['price']
                cost = ((usage['prompt_tokens'] - usage['cached_tokens']) * p['input'] + usage['cached_tokens'] * p.get('cached', p['input'])
                        + usage['completion_tokens'] * p['output']) / 1e6
            else:
                cost = pricing.cost(candidate['model'], usage['prompt_tokens'], usage['completion_tokens'], usage['cached_tokens'])
            budget.charge(cost)
        scenario = item['tags'].get('scenario', 'single')
        record.update({
            'request_params': request['params'],
            'content': result['content'], 'tool_calls': result['tool_calls'],
            'ttft_s': result['ttft_s'], 'total_s': result['total_s'], 'finish_reason': result['finish_reason'],
            'usage': result['usage'], 'cost_usd': round(cost, 6) if cost is not None else None, 'error': result['error'],
            'voice': voice_metrics(result['content'], scenario, result['tool_calls']),
            'action_check': action_check(item.get('expected'), result['tool_calls']),
        })
        for key in ('mode', 'speech_ttft_s', 'followup', 'waits_for_game', 'calls'):
            if key in result:
                record[key] = result[key]
        with lock:
            summary['errors' if result['error'] else 'done'] += 1
            append_jsonl(out_path, record)
            status = f"error: {result['error'][:80]}" if result['error'] else f"ttft {result['ttft_s']}s total {result['total_s']}s"
            log(f"  [{summary['done'] + summary['errors']}/{len(jobs)}] {candidate['label']} {item['id']}: {status}")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(work, jobs))
    summary['spent_usd'] = round(budget.spent, 4)
    return summary
