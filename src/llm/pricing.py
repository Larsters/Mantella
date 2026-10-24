"""LLM price table (USD per 1M tokens) and cost calculation.

Shared by the Control Center spend tab and the eval toolkit. Prices live in data/llm_pricing.json.
"""
import json
import os
from functools import lru_cache

PRICING_FILE = os.path.join('data', 'llm_pricing.json')


@lru_cache(maxsize=1)
def load_pricing(path: str = PRICING_FILE) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def price_for(model: str | None, path: str = PRICING_FILE) -> dict | None:
    """Exact match first, then the longest known prefix (e.g. dated snapshots); OpenRouter ids are stripped."""
    models = load_pricing(path)['models']
    name = (model or '').split('/')[-1]
    if name in models:
        return models[name]
    matches = [m for m in models if name.startswith(m)]
    return models[max(matches, key=len)] if matches else None


def cost_usd(model: str | None, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0,
             path: str = PRICING_FILE) -> float | None:
    """None when the model has no known price."""
    price = price_for(model, path)
    if price is None:
        return None
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (uncached * price['input'] + cached_tokens * price['cached'] + completion_tokens * price['output']) / 1e6


def eur_per_usd(path: str = PRICING_FILE) -> float:
    return load_pricing(path).get('eur_per_usd', 1.0)
