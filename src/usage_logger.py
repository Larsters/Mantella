"""Appends one JSON line per LLM request to usage.jsonl in the Mantella user folder.

Read by tools/mantella-usage (dashboard) in the skyrim-mantella-mac project.
Never raises: a logging failure must not break a conversation.
"""
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

import src.utils as utils

logger = utils.get_logger()

_lock = threading.Lock()
_context: dict[str, Any] = {}
_log_path: str | None = None

_CALL_TYPES = {
    'LLMClient': 'dialogue',
    'SummaryLLMClient': 'memory',
    'ImageClient': 'vision',
    'FunctionClient': 'actions',
}


def _get_log_path() -> str:
    global _log_path
    if _log_path:
        return _log_path
    path = os.environ.get('MANTELLA_USAGE_LOG')
    if not path:
        folder = None
        for handler in logging.getLogger().handlers + logger.handlers:
            if isinstance(handler, logging.FileHandler):
                folder = os.path.dirname(handler.baseFilename)
                break
        if not folder:
            folder = os.path.join(os.path.expanduser('~'), 'Documents', 'My Games', 'Mantella')
        path = os.path.join(folder, 'usage.jsonl')
    _log_path = path
    return path


def new_conversation() -> None:
    _context.clear()
    _context['conversation_id'] = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]


def set_context(**values: Any) -> None:
    _context.update(values)


def call_type_for(client: object) -> str:
    return _CALL_TYPES.get(type(client).__name__, type(client).__name__)


def log(call_type: str, model: str, base_url: str, usage: Any, latency_s: float,
        estimated: bool = False, **extra: Any) -> None:
    try:
        if usage is None:
            return
        get = (lambda k: usage.get(k)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
        details = get('prompt_tokens_details')
        cached = 0
        if details is not None:
            cached = (details.get('cached_tokens') if isinstance(details, dict) else getattr(details, 'cached_tokens', 0)) or 0
        record = {
            'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'conversation_id': _context.get('conversation_id'),
            'call_type': call_type,
            'model': model,
            'base_url': base_url,
            'npcs': _context.get('npcs'),
            'speaker': _context.get('speaker'),
            'prompt_tokens': get('prompt_tokens') or 0,
            'cached_tokens': cached,
            'completion_tokens': get('completion_tokens') or 0,
            'latency_s': round(latency_s, 3),
            'estimated': estimated,
        }
        record.update(extra)
        with _lock, open(_get_log_path(), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.debug(f'Usage logging failed: {e}')
