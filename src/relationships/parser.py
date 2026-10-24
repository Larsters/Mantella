"""Robust parsing of the relationship-update reply.

Structured output (json_schema) is requested, but many OpenAI-compatible providers ignore it or
the call falls back to plain text, so the reply may contain code fences, prose around the JSON,
trailing commas, single quotes or a bare list. This module tries hard to recover a dict of the
form {"npcs": [ {...update for one NPC...}, ... ]} and returns None when nothing usable is found.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any


def _balanced_json_candidates(text: str) -> list[str]:
    """Return substrings that start with { or [ and end at the matching bracket (string-aware), in order."""
    candidates = []
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        stack = []
        in_string: str | None = None
        escaped = False
        for i in range(start, len(text)):
            c = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == in_string:
                    in_string = None
                continue
            if c in "\"'":
                in_string = c
            elif c in "{[":
                stack.append("}" if c == "{" else "]")
            elif c in "}]":
                if not stack or stack.pop() != c:
                    break
                if not stack:
                    candidates.append(text[start:i + 1])
                    break
        if len(candidates) >= 20:
            break
    return candidates


def _loads_lenient(snippet: str) -> Any:
    try:
        return json.loads(snippet)
    except (json.JSONDecodeError, ValueError):
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", snippet)  # trailing commas
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass
    # Python-literal style (single quotes, True/False/None)
    try:
        pyish = repaired.replace("true", "True").replace("false", "False").replace("null", "None")
        return ast.literal_eval(pyish)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _coerce(data: Any) -> dict[str, Any] | None:
    if isinstance(data, list):
        data = {"npcs": data}
    if not isinstance(data, dict):
        return None
    if "npcs" not in data:
        # A single NPC update without the wrapper
        keys = {"opinion_change", "new_facts", "new_grudges", "attitude", "reason"}
        if keys & set(data.keys()):
            data = {"npcs": [data]}
        else:
            return None
    npcs = data.get("npcs")
    if isinstance(npcs, dict):
        # {"npcs": {"Lydia": {...}}}
        npcs = [{"name": name, **value} for name, value in npcs.items() if isinstance(value, dict)]
    if not isinstance(npcs, list):
        return None
    data["npcs"] = [n for n in npcs if isinstance(n, dict)]
    return data


def parse_update_reply(reply: str | None) -> dict[str, Any] | None:
    if not reply or not isinstance(reply, str):
        return None
    text = reply.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    parsed = _coerce(_loads_lenient(text))
    if parsed is not None:
        return parsed
    for candidate in _balanced_json_candidates(text):
        parsed = _coerce(_loads_lenient(candidate))
        if parsed is not None:
            return parsed
    return None
