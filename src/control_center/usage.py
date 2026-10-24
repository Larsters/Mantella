"""Reads usage.jsonl (written by src/usage_logger.py) and prices each request with src.llm.pricing."""
from __future__ import annotations

import json
import os
import threading
from typing import Any

import src.usage_logger as usage_logger
from src.llm import pricing

_lock = threading.Lock()
_cache: dict[str, Any] = {'key': None, 'records': []}


def log_path() -> str:
    return usage_logger._get_log_path()


def known_models() -> list[str]:
    try:
        return list(pricing.load_pricing()['models'].keys())
    except Exception:
        return []


def eur_per_usd() -> float:
    try:
        return pricing.eur_per_usd()
    except Exception:
        return 1.0


def cost_usd(record: dict) -> float | None:
    try:
        return pricing.cost_usd(record.get('model'), record.get('prompt_tokens') or 0, record.get('completion_tokens') or 0,
                                record.get('cached_tokens') or 0)
    except Exception:
        return None


def load_records(path: str | None = None) -> list[dict]:
    """All records with cost_usd added. Re-reads the file only when its size or mtime changed."""
    path = path or log_path()
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (path, st.st_size, st.st_mtime_ns)
    with _lock:
        if _cache['key'] == key:
            return _cache['records']
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict):
                continue
            r['cost_usd'] = cost_usd(r)
            records.append(r)
    with _lock:
        _cache['key'] = key
        _cache['records'] = records
    return records


def usage_payload() -> dict:
    return {'records': load_records(), 'eur_per_usd': eur_per_usd(), 'log_path': log_path()}
