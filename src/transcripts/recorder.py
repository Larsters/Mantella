"""Opt-in transcript recorder for dialogue LLM calls (W9: conversation capture & model evals).

When `save_conversation_transcripts` is on, every dialogue LLM call (LLMClient.streaming_call) is
appended as one JSON line to `<Mantella user folder>/data/transcripts/<conversation_id>.jsonl`:
the exact request (messages incl. system prompt, tools, params, model) and the response (text, tool
calls, time to first token, total time, usage). The conversation id is the same one usage.jsonl uses
(src/usage_logger.py), so the two logs can be joined.

Base64 images are never written: they are replaced with a placeholder and the record is flagged.
The offline eval toolkit (`python -m src.evals`) reads these files.

Never raises: a recording failure must not break a conversation.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from typing import Any

import src.utils as utils
import src.usage_logger as usage_logger

logger = utils.get_logger()

SCHEMA_VERSION = 1
RECORDED_CALL_TYPES = {'dialogue'}

_lock = threading.Lock()
_enabled: bool = False
_folder: str | None = None
_turn_counters: dict[str, int] = {}


def configure(enabled: bool, save_folder: str | None) -> None:
    """Called by the config loader whenever config values are (re)loaded."""
    global _enabled, _folder
    _enabled = bool(enabled)
    if save_folder:
        _folder = os.path.join(save_folder, 'data', 'transcripts')
    if _enabled:
        logger.log(23, f'Conversation transcripts are saved to {transcripts_folder()}')


def is_enabled() -> bool:
    return _enabled


def transcripts_folder() -> str:
    if _folder:
        return _folder
    return os.path.join(os.path.expanduser('~'), 'Documents', 'My Games', 'Mantella', 'data', 'transcripts')


def strip_images(messages: list[Any]) -> tuple[list[Any], int]:
    """Returns a deep copy of the messages with base64 image data replaced by a placeholder,
    and the number of images that were stripped. Remote image URLs are kept."""
    stripped = 0
    result = copy.deepcopy(messages)
    for message in result:
        if not isinstance(message, dict) or not isinstance(message.get('content'), list):
            continue
        for part in message['content']:
            if not isinstance(part, dict) or part.get('type') != 'image_url':
                continue
            image = part.get('image_url')
            url = image.get('url', '') if isinstance(image, dict) else str(image or '')
            if url.startswith('data:'):
                stripped += 1
                placeholder = f'<image omitted: {len(url)} chars base64>'
                if isinstance(image, dict):
                    image['url'] = placeholder
                else:
                    part['image_url'] = {'url': placeholder}
                part['omitted'] = True
    return result, stripped


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    get = (lambda k: usage.get(k)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
    details = get('prompt_tokens_details')
    cached = 0
    if details is not None:
        cached = (details.get('cached_tokens') if isinstance(details, dict) else getattr(details, 'cached_tokens', 0)) or 0
    completion_details = get('completion_tokens_details')
    reasoning = 0
    if completion_details is not None:
        reasoning = (completion_details.get('reasoning_tokens') if isinstance(completion_details, dict) else getattr(completion_details, 'reasoning_tokens', 0)) or 0
    return {
        'prompt_tokens': get('prompt_tokens') or 0,
        'cached_tokens': cached,
        'completion_tokens': get('completion_tokens') or 0,
        'reasoning_tokens': reasoning,
    }


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return json.loads(json.dumps(value, default=lambda o: getattr(o, '__dict__', str(o))))


class _NullTurn:
    """Returned when recording is off: every method is a no-op."""
    completed = False

    def mark_first_token(self) -> None:
        pass

    def fail(self, error: BaseException) -> None:
        pass

    def finish(self, *args: Any, **kwargs: Any) -> None:
        pass


NULL_TURN = _NullTurn()


class TranscriptTurn:
    def __init__(self, call_type: str, model: str, base_url: str, messages: list[Any],
                 request_params: dict[str, Any], is_multi_npc: bool) -> None:
        self.start_time = time.time()
        self.first_token_time: float | None = None
        self.completed = False
        self.error: str | None = None
        context = dict(usage_logger._context)
        self.conversation_id: str = context.get('conversation_id') or time.strftime('%Y%m%d-%H%M%S-') + 'nocontext'
        clean_messages, images = strip_images(messages)
        params = {k: v for k, v in (request_params or {}).items() if k not in ('tools', 'stream_options')}
        tools = (request_params or {}).get('tools')
        self.record: dict[str, Any] = {
            'schema': SCHEMA_VERSION,
            'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'conversation_id': self.conversation_id,
            'call_type': call_type,
            'model': model,
            'base_url': base_url,
            'speaker': context.get('speaker'),
            'npcs': context.get('npcs'),
            'has_player': context.get('has_player'),
            'is_multi_npc': is_multi_npc,
            'request': {
                'model': model,
                'messages': _jsonable(clean_messages),
                'tools': _jsonable(copy.deepcopy(tools)) if tools else None,
                'params': _jsonable(copy.deepcopy(params)),
            },
            'flags': {
                'images_stripped': images,
                'followup_after_tool': _is_followup_after_tool(clean_messages, tools),
            },
        }

    def mark_first_token(self) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def fail(self, error: BaseException) -> None:
        self.error = f'{type(error).__name__}: {error}'

    def finish(self, content: str, tool_calls: list[dict] | None, usage: Any, estimated: bool) -> None:
        try:
            end = time.time()
            self.record['response'] = {
                'content': content or '',
                'tool_calls': _jsonable(copy.deepcopy(tool_calls)) if tool_calls else None,
                'completed': self.completed,
                'error': self.error,
            }
            self.record['timing'] = {
                'ttft_s': round(self.first_token_time - self.start_time, 3) if self.first_token_time else None,
                'total_s': round(end - self.start_time, 3),
            }
            usage_record = _usage_dict(usage)
            if usage_record is not None:
                usage_record['estimated'] = estimated
            self.record['usage'] = usage_record
            _write(self.record)
        except Exception as e:
            logger.debug(f'Transcript recording failed: {e}')


def _is_followup_after_tool(messages: list[Any], tools: Any) -> bool:
    """True for Mantella's second call (no tools) that asks for the spoken line after a tool-only reply."""
    if tools:
        return False
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get('role') == 'tool' or (message.get('role') == 'assistant' and message.get('tool_calls')):
            return True
        if message.get('role') == 'user':
            return False
    return False


def _write(record: dict[str, Any]) -> None:
    folder = transcripts_folder()
    with _lock:
        conversation_id = record['conversation_id']
        record['turn_index'] = _turn_counters.get(conversation_id, 0)
        _turn_counters[conversation_id] = record['turn_index'] + 1
        os.makedirs(folder, exist_ok=True)
        safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in conversation_id)
        with open(os.path.join(folder, f'{safe_name}.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def start_turn(client: object, model: str, base_url: str, messages: list[Any],
               request_params: dict[str, Any] | None, is_multi_npc: bool) -> TranscriptTurn | _NullTurn:
    """Starts timing one LLM call. Returns a no-op object when recording is off or the call isn't dialogue."""
    if not _enabled:
        return NULL_TURN
    try:
        call_type = usage_logger.call_type_for(client)
        if call_type not in RECORDED_CALL_TYPES:
            return NULL_TURN
        return TranscriptTurn(call_type, model, base_url, messages, request_params or {}, is_multi_npc)
    except Exception as e:
        logger.debug(f'Transcript recording failed: {e}')
        return NULL_TURN
