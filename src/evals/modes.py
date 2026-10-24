"""Turn modes: replay one player line the way Mantella's server turns it into speech and actions.

A candidate picks the mode with `"mode"`:
- absent (default, unchanged): one streaming call with the item's tools; the reply is whatever comes back.
- `"single"` (`custom_function_model = False`): the main model gets the tools. If it answers with tool calls only and
  none of them waits for the game (`requires_response`), a second call without tools follows, with the tool calls
  and "done" results in the history (output_manager: "Making second LLM call for text response...").
- `"tool_model"` (`custom_function_model = True`): first a non-streaming call to the tool model
  (`candidate["tool_model"] = {"model", "params"}`), whose system prompt is `item["function_system"]`
  (function_llm_prompt + W12 rules + Scene, see `w12.py`). Then the main model streams WITHOUT tools and without
  seeing the chosen tool calls (client_base builds its messages before the tool model runs). If that main reply is
  empty, the same second call as above follows.

Timing is wall clock from the start of the turn: `ttft_s` is the first spoken token (for tool-only turns that wait
for the game: the moment the action is known), `total_s` the end of the last call. Cost sums every call.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Callable

from src.evals.common import Pricing, adapt_request_params
from src.evals.llm import DEFAULT_BASE_URL, complete_chat, stream_chat

MODES = ('single', 'tool_model')
DEFAULT_TOOL_MODEL_PARAMS = {'max_tokens': 500, 'stop': ['#']}  # upstream function_llm_params
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def actions_waiting_for_game(repo_root: str = REPO_ROOT) -> set[str]:
    """Tool names whose action has `requires_response` (the server skips the second call for them)."""
    names = set()
    for path in glob.glob(os.path.join(repo_root, 'data', 'actions', '*.json')):
        with open(path, encoding='utf-8') as f:
            action = json.load(f)
        if action.get('requires_response'):
            names.add(action.get('name', ''))
    return names


def _cost(pricing: Pricing, model: str, usage: dict[str, int] | None) -> float:
    if not usage:
        return 0.0
    return pricing.cost(model, usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), usage.get('cached_tokens', 0))


def _sum_usage(usages: list[dict[str, int] | None]) -> dict[str, int] | None:
    present = [u for u in usages if u]
    if not present:
        return None
    return {k: sum(u.get(k, 0) for u in present) for k in ('prompt_tokens', 'cached_tokens', 'completion_tokens', 'reasoning_tokens')}


def _followup_messages(messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What output_manager._add_tool_calls_to_history adds before the second call."""
    calls = [{'id': c.get('id') or f'call_{i}', 'type': 'function', 'function': c['function']} for i, c in enumerate(tool_calls)]
    return (messages + [{'role': 'assistant', 'content': None, 'tool_calls': calls}]
            + [{'role': 'tool', 'tool_call_id': c['id'], 'content': 'done'} for c in calls])


def tool_model_request(item: dict[str, Any], candidate: dict[str, Any], request: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """(model, messages, params incl. tools) for the tool model: same thread, system prompt replaced."""
    spec = candidate.get('tool_model') or {}
    model = spec.get('model', 'gpt-4o-mini')
    system = item.get('function_system')
    if not system:
        raise ValueError('tool_model mode needs item["function_system"] (build the set with --trust)')
    messages = [m for m in request['messages'] if m.get('role') != 'system']
    messages = [{'role': 'system', 'content': system}] + [_text_only(m) for m in messages]
    params = dict(DEFAULT_TOOL_MODEL_PARAMS)
    for key, value in (spec.get('params') or {}).items():
        if value is None:
            params.pop(key, None)
        else:
            params[key] = value
    params = adapt_request_params(model, params)
    if request.get('tools'):
        params['tools'] = request['tools']
    return model, messages, params


def _text_only(message: dict[str, Any]) -> dict[str, Any]:
    """The tool model never gets the screenshot (only the main call does)."""
    if isinstance(message.get('content'), list):
        text = ' '.join(p.get('text', '') for p in message['content'] if isinstance(p, dict) and p.get('type') == 'text')
        return {**message, 'content': text}
    return message


def run_turn(item: dict[str, Any], candidate: dict[str, Any], request: dict[str, Any],
             client_factory: Callable[[str, str | None], Any], pricing: Pricing,
             waiting_actions: set[str] | None = None) -> dict[str, Any]:
    """Runs one turn in the candidate's mode. Returns the fields replay stores for a reply."""
    mode = candidate.get('mode')
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")
    waiting = actions_waiting_for_game() if waiting_actions is None else waiting_actions
    base_url = candidate.get('base_url', DEFAULT_BASE_URL)
    main_client = client_factory(base_url, candidate.get('api_key_env'))
    calls: list[dict[str, Any]] = []
    start = time.perf_counter()
    tool_calls: list[dict[str, Any]] | None = None
    error = None

    if mode == 'tool_model':
        tm_model, tm_messages, tm_params = tool_model_request(item, candidate, request)
        tm_client = client_factory((candidate.get('tool_model') or {}).get('base_url', DEFAULT_BASE_URL), None)
        tm = complete_chat(tm_client, tm_model, tm_messages, tm_params)
        calls.append({'role': 'tool_model', 'model': tm_model, 'total_s': tm['total_s'], 'usage': tm['usage'],
                      'tool_calls': [c['function']['name'] for c in tm.get('tool_calls') or []], 'error': tm['error'],
                      'cost_usd': _cost(pricing, tm_model, tm['usage'])})
        tool_calls = tm.get('tool_calls') or None  # an error here = no actions this turn, like the server
        offset = time.perf_counter() - start
        main = stream_chat(main_client, request['model'], request['messages'], None, request['params'])
    else:
        offset = 0.0
        main = stream_chat(main_client, request['model'], request['messages'], request.get('tools'), request['params'])
        tool_calls = main['tool_calls']
    calls.append({'role': 'main', 'model': request['model'], 'ttft_s': main['ttft_s'], 'total_s': main['total_s'],
                  'usage': main['usage'], 'error': main['error'], 'cost_usd': _cost(pricing, request['model'], main['usage'])})
    error = main['error']
    content = main['content'] or ''
    first_speech = offset + main['ttft_s'] if (content.strip() and main['ttft_s'] is not None) else None
    first_action = (calls[0]['total_s'] if mode == 'tool_model' else (main['ttft_s'] and offset + main['ttft_s'])) if tool_calls else None
    finish_reason = main['finish_reason']

    waits_for_game = bool(tool_calls) and any(c['function']['name'] in waiting for c in tool_calls)
    followup = bool(tool_calls) and not content.strip() and not waits_for_game and not error
    if followup:
        # no tools now: the server's adapter may pick another effort (e.g. llm_params "minimal" -> low on gpt-5.6-luna)
        params = adapt_request_params(request['model'], dict(request.get('raw_params') or request['params']))
        offset2 = time.perf_counter() - start
        second = stream_chat(main_client, request['model'], _followup_messages(request['messages'], tool_calls), None, params)
        calls.append({'role': 'followup', 'model': request['model'], 'ttft_s': second['ttft_s'], 'total_s': second['total_s'],
                      'usage': second['usage'], 'error': second['error'], 'request_params': params,
                      'cost_usd': _cost(pricing, request['model'], second['usage'])})
        content = second['content'] or ''
        error = second['error']
        finish_reason = second['finish_reason']
        if content.strip() and second['ttft_s'] is not None:
            first_speech = offset2 + second['ttft_s']
    total = round(time.perf_counter() - start, 3)
    ttft = first_speech if first_speech is not None else first_action
    return {
        'content': content, 'tool_calls': tool_calls, 'error': error, 'finish_reason': finish_reason,
        'ttft_s': round(ttft, 3) if ttft is not None else None,
        'speech_ttft_s': round(first_speech, 3) if first_speech is not None else None,
        'total_s': total, 'usage': _sum_usage([c['usage'] for c in calls]),
        'cost_usd': sum(c['cost_usd'] for c in calls), 'mode': mode, 'calls': calls,
        'followup': followup, 'waits_for_game': waits_for_game,
    }


def estimate_extra_calls(candidate: dict[str, Any], input_tokens: int, tool_tokens: int, main_cost: tuple[float, float],
                         pricing: Pricing) -> tuple[int, float, float]:
    """Extra (calls, expected USD, worst USD) per turn on top of the main call: the tool-model call and the
    possible second call (expected in ~1 of 3 turns in single mode, worst case every turn)."""
    mode = candidate.get('mode')
    if mode not in MODES:
        return 0, 0.0, 0.0
    calls, expected, worst = 1, main_cost[0] * (0.35 if mode == 'single' else 0.05), main_cost[1]
    if mode == 'tool_model':
        spec = candidate.get('tool_model') or {}
        price, _ = pricing.lookup(spec.get('model', 'gpt-4o-mini'))
        cap = int((spec.get('params') or {}).get('max_tokens', DEFAULT_TOOL_MODEL_PARAMS['max_tokens']))
        tm_in = input_tokens + tool_tokens
        calls += 1
        expected += (tm_in * price['input'] + 40 * price['output']) / 1e6
        worst += (tm_in * price['input'] + cap * price['output']) / 1e6
    return calls, expected, worst
