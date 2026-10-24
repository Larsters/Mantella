"""Deterministic checks computed without a judge: voice heuristics and expected/forbidden tool rules."""
from __future__ import annotations

import re
from typing import Any

MAX_SENTENCES = {'single': 4, 'multi': 12, 'radiant': None}  # Mantella's default max_count_sentences parser limits
_SENTENCE_END = re.compile(r'[.!?]+(?:\s|$)')
_MARKDOWN = re.compile(r'(\*\*|__|`|^#+\s|^\s*[-*]\s|\[[^\]]*\]\()', re.MULTILINE)
_NAME_PREFIX = re.compile(r"^\s*[A-Z][\w' -]{0,40}:\s", re.MULTILINE)


def voice_metrics(text: str, scenario: str, tool_calls: list[dict] | None) -> dict[str, Any]:
    text = text or ''
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    limit = MAX_SENTENCES.get(scenario)
    return {
        'words': len(text.split()),
        'sentences': len(sentences),
        'markdown': bool(_MARKDOWN.search(text)),
        'asterisks': '*' in text,
        'quotes': '"' in text or '“' in text,
        # in single-NPC replies any "Word: " prefix makes Mantella discard the rest of the reply
        'name_prefix_in_single': scenario == 'single' and bool(_NAME_PREFIX.search(text)),
        'over_sentence_limit': bool(limit and len(sentences) > limit),
        'tool_only': not text.strip() and bool(tool_calls),
    }


def tool_names(tool_calls: list[dict] | None) -> list[str]:
    return [c.get('function', {}).get('name', '') for c in tool_calls or []]


def action_check(expected: dict[str, Any] | None, tool_calls: list[dict] | None) -> dict[str, Any] | None:
    """None when the item has no expectations (real transcripts)."""
    if not expected or ('expect_tools' not in expected and 'forbid_tools' not in expected):
        return None
    called = tool_names(tool_calls)
    expect = expected.get('expect_tools') or []
    forbid = expected.get('forbid_tools') or []
    expected_ok = (not expect) or any(name in called for name in expect)
    forbidden_called = [name for name in called if name in forbid]
    return {'called': called, 'expected_ok': expected_ok, 'forbidden_called': forbidden_called,
            'pass': expected_ok and not forbidden_called}
