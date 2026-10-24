"""Shared helpers for the eval toolkit: paths, JSONL I/O, pricing, request-param adaptation, token estimates.

The toolkit is offline/CLI only and deliberately avoids importing Mantella's runtime (no game, no config.ini).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, Iterator

import src.llm.pricing as shared_pricing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRICING_FILE = os.path.join(REPO_ROOT, shared_pricing.PRICING_FILE)  # absolute, so the CLI works from any cwd


# ---------------------------------------------------------------- paths

def default_data_dir() -> str:
    """`~/Documents/My Games/Mantella/data` unless MANTELLA_DATA_DIR is set."""
    env = os.environ.get('MANTELLA_DATA_DIR')
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser('~'), 'Documents', 'My Games', 'Mantella', 'data')


def transcripts_dir(data_dir: str) -> str:
    return os.path.join(data_dir, 'transcripts')


def evals_dir(data_dir: str) -> str:
    return os.path.join(data_dir, 'evals')


def sets_dir(data_dir: str) -> str:
    return os.path.join(evals_dir(data_dir), 'sets')


def set_path(data_dir: str, name_or_path: str) -> str:
    if os.path.sep in name_or_path or name_or_path.endswith('.jsonl'):
        return os.path.expanduser(name_or_path)
    return os.path.join(sets_dir(data_dir), f'{name_or_path}.jsonl')


def run_dir(data_dir: str, run_id: str) -> str:
    if os.path.sep in run_id:
        return os.path.expanduser(run_id)
    return os.path.join(evals_dir(data_dir), run_id)


# ---------------------------------------------------------------- JSONL

def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f'warning: skipping malformed line {line_number} in {path}')


def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    return count


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------- pricing

class Pricing:
    """Thin wrapper over the shared src.llm.pricing table (data/llm_pricing.json, USD per 1M tokens).
    Unknown models are priced as the most expensive listed model, so budget checks stay conservative."""

    def __init__(self, path: str = PRICING_FILE, overrides: dict[str, dict[str, float]] | None = None) -> None:
        self.path = path
        self.overrides = overrides or {}

    def lookup(self, model: str) -> tuple[dict[str, float], bool]:
        """Returns (price, known)."""
        if model in self.overrides:
            return self.overrides[model], True
        price = shared_pricing.price_for(model, self.path)
        if price is not None:
            return price, True
        models = shared_pricing.load_pricing(self.path)['models']
        return max(models.values(), key=lambda p: p['input'] + p['output']), False

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        if model not in self.overrides:
            known = shared_pricing.cost_usd(model, prompt_tokens, completion_tokens, cached_tokens, self.path)
            if known is not None:
                return known
        price, _ = self.lookup(model)
        uncached = max(prompt_tokens - cached_tokens, 0)
        return (uncached * price['input'] + cached_tokens * price.get('cached', price['input'])
                + completion_tokens * price['output']) / 1_000_000


# ---------------------------------------------------------------- request params

_REASONING_MODEL = re.compile(r'^(o\d|gpt-5)', re.IGNORECASE)
_SAMPLING_PARAMS = ('temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'logit_bias', 'stop', 'logprobs', 'top_logprobs')


def _fallback_adapt(model: str, params: dict[str, Any]) -> dict[str, Any]:
    """Minimal stand-in used only when W3's src.llm.model_capabilities is not available:
    reasoning families (o*, gpt-5*) get max_completion_tokens and no sampling params; other models
    lose reasoning-only params. Everything else is passed through unchanged."""
    params = dict(params)
    name = model.split('/')[-1]
    if _REASONING_MODEL.match(name):
        if 'max_tokens' in params:
            params.setdefault('max_completion_tokens', params.pop('max_tokens'))
        for key in _SAMPLING_PARAMS:
            params.pop(key, None)
    else:
        params.pop('reasoning_effort', None)
        params.pop('verbosity', None)
    return params


def adapt_request_params(model: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        from src.llm.model_capabilities import adapt_request_params as w3_adapt  # provided by feat/openai-latest-models
    except Exception:
        return _fallback_adapt(model, params)
    return w3_adapt(model, dict(params))


def is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL.match(model.split('/')[-1]))


def output_cap(params: dict[str, Any], default: int) -> int:
    for key in ('max_completion_tokens', 'max_tokens', 'max_output_tokens'):
        if params.get(key):
            return int(params[key])
    return default


REASONING_ALLOWANCE = {'none': 0, 'minimal': 0, 'low': 300, 'medium': 1200, 'high': 4000, 'xhigh': 8000}


def reasoning_allowance(model: str, params: dict[str, Any]) -> int:
    """Expected hidden reasoning tokens per reply (a guess used only for cost estimates)."""
    if not is_reasoning_model(model):
        return 0
    return REASONING_ALLOWANCE.get(str(params.get('reasoning_effort', 'medium')).lower(), 1200)


# ---------------------------------------------------------------- token estimates

_ENCODER: Any = None


def _encoder() -> Any:
    """tiktoken's cl100k_base when installed (o200k models tokenize slightly tighter, so this stays conservative)."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding('cl100k_base')
        except Exception:
            _ENCODER = False
    return _ENCODER


def estimate_tokens(value: Any) -> int:
    """Token count for cost estimates: tiktoken when available, else ≈3.5 characters per token.
    Actual usage always comes from the API."""
    if value is None:
        return 0
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    encoder = _encoder()
    if encoder:
        return len(encoder.encode(text, disallowed_special=())) + 1
    return int(len(text) / 3.5) + 1


def estimate_request_tokens(messages: list[Any], tools: list[Any] | None) -> int:
    return estimate_tokens(messages) + estimate_tokens(tools) + 4 * len(messages)


# ---------------------------------------------------------------- misc

def message_text(message: dict[str, Any]) -> str:
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                parts.append(part.get('text', ''))
            elif isinstance(part, dict) and part.get('type') == 'image_url':
                parts.append('[screenshot]')
        return ' '.join(parts)
    return ''
