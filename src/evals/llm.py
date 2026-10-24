"""OpenAI-compatible client helpers with timing (time to first token, total time) and usage capture."""
from __future__ import annotations

import os
import time
from typing import Any, Callable

DEFAULT_BASE_URL = 'https://api.openai.com/v1'
KEY_FILE_CANDIDATES = (
    'GPT_SECRET_KEY.txt',
    os.path.join(os.path.expanduser('~'), 'Games', 'mantella-server', 'GPT_SECRET_KEY.txt'),
)

ClientFactory = Callable[[str, str], Any]  # (base_url, api_key) -> OpenAI-like client


def find_api_key(api_key_env: str | None = None, key_file: str | None = None) -> str | None:
    """Looks up an API key without ever printing it."""
    for env in (api_key_env, 'OPENAI_API_KEY'):
        if env and os.environ.get(env):
            return os.environ[env].strip()
    for path in ((key_file,) if key_file else KEY_FILE_CANDIDATES):
        if path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), encoding='utf-8') as f:
                key = f.read().strip()
            if key:
                return key
    return None


def openai_client_factory(key_file: str | None = None) -> ClientFactory:
    cache: dict[tuple[str, str | None], Any] = {}

    def factory(base_url: str, api_key_env: str | None = None) -> Any:
        from openai import OpenAI
        cache_key = (base_url, api_key_env)
        if cache_key not in cache:
            key = find_api_key(api_key_env, key_file)
            if not key:
                raise RuntimeError('No API key found (set OPENAI_API_KEY, the candidate\'s api_key_env, or --key-file).')
            cache[cache_key] = OpenAI(api_key=key, base_url=base_url, max_retries=2, timeout=120)
        return cache[cache_key]

    return factory


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def usage_to_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    details = _get(usage, 'prompt_tokens_details')
    completion_details = _get(usage, 'completion_tokens_details')
    return {
        'prompt_tokens': _get(usage, 'prompt_tokens', 0) or 0,
        'cached_tokens': _get(details, 'cached_tokens', 0) or 0,
        'completion_tokens': _get(usage, 'completion_tokens', 0) or 0,
        'reasoning_tokens': _get(completion_details, 'reasoning_tokens', 0) or 0,
    }


def stream_chat(client: Any, model: str, messages: list[Any], tools: list[Any] | None,
                params: dict[str, Any], include_usage: bool = True) -> dict[str, Any]:
    """One streaming chat completion, measured the same way the server measures it:
    TTFT = first content or tool-call delta, total = last chunk."""
    request: dict[str, Any] = dict(params)
    if tools:
        request['tools'] = tools
    if include_usage:
        request['stream_options'] = {'include_usage': True}
    result: dict[str, Any] = {'content': '', 'tool_calls': None, 'ttft_s': None, 'total_s': None,
                              'usage': None, 'finish_reason': None, 'error': None}
    tool_calls: dict[int, dict[str, Any]] = {}
    start = time.perf_counter()
    try:
        stream = client.chat.completions.create(model=model, messages=messages, stream=True, **request)
        for chunk in stream:
            if _get(chunk, 'usage'):
                result['usage'] = usage_to_dict(_get(chunk, 'usage'))
            choices = _get(chunk, 'choices') or []
            if not choices:
                continue
            choice = choices[0]
            if _get(choice, 'finish_reason'):
                result['finish_reason'] = _get(choice, 'finish_reason')
            delta = _get(choice, 'delta')
            if delta is None:
                continue
            content = _get(delta, 'content')
            delta_tools = _get(delta, 'tool_calls')
            if (content or delta_tools) and result['ttft_s'] is None:
                result['ttft_s'] = round(time.perf_counter() - start, 3)
            if content:
                result['content'] += content
            for call in delta_tools or []:
                index = _get(call, 'index', 0) or 0
                entry = tool_calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if _get(call, 'id'):
                    entry['id'] = _get(call, 'id')
                function = _get(call, 'function')
                if _get(function, 'name'):
                    entry['function']['name'] += _get(function, 'name')
                if _get(function, 'arguments'):
                    entry['function']['arguments'] += _get(function, 'arguments')
    except Exception as e:  # recorded, never raised: one failing call must not end a run
        result['error'] = f'{type(e).__name__}: {e}'
    result['total_s'] = round(time.perf_counter() - start, 3)
    if tool_calls:
        result['tool_calls'] = [tool_calls[i] for i in sorted(tool_calls)]
    return result


def complete_chat(client: Any, model: str, messages: list[Any], params: dict[str, Any]) -> dict[str, Any]:
    """One non-streaming chat completion (used by the judge)."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(model=model, messages=messages, **params)
        choices = _get(response, 'choices') or []
        message = _get(choices[0], 'message') if choices else None
        return {'content': _get(message, 'content') or '', 'usage': usage_to_dict(_get(response, 'usage')),
                'total_s': round(time.perf_counter() - start, 3), 'error': None}
    except Exception as e:
        return {'content': '', 'usage': None, 'total_s': round(time.perf_counter() - start, 3),
                'error': f'{type(e).__name__}: {e}'}
