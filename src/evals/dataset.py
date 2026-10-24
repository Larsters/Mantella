"""Eval sets: select recorded turns from transcripts (dedupe + tags) and/or the seed scenarios."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from typing import Any

from src.evals.common import message_text, read_jsonl, transcripts_dir


def scenario_of(record: dict[str, Any]) -> str:
    if not record.get('is_multi_npc'):
        return 'single'
    if record.get('has_player') is False:
        return 'radiant'
    if record.get('has_player') is None:
        # older records without has_player: the radiant prompt never mentions "(the player)"
        system = next((message_text(m) for m in record['request']['messages'] if m.get('role') == 'system'), '')
        return 'multi' if '(the player)' in system else 'radiant'
    return 'multi'


def turn_from_record(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get('response') or {}
    request = record['request']
    tool_calls = response.get('tool_calls') or []
    return {
        'id': f"{record['conversation_id']}#{record.get('turn_index', 0)}",
        'source': 'transcript',
        'tags': {
            'scenario': scenario_of(record),
            'npcs': record.get('npcs'),
            'speaker': record.get('speaker'),
            'tools_offered': bool(request.get('tools')),
            'had_tool_call': bool(tool_calls),
            'original_tools_called': [c['function']['name'] for c in tool_calls if c.get('function')],
            'followup_after_tool': bool((record.get('flags') or {}).get('followup_after_tool')),
            'had_image': bool((record.get('flags') or {}).get('images_stripped')),
            'expects_tool': None,
        },
        'request': request,
        'expected': None,
        'prompt_vars': record.get('prompt_vars'),
        'original_response': {
            'model': record.get('model'),
            'content': response.get('content'),
            'tool_calls': tool_calls or None,
            'timing': record.get('timing'),
            'usage': record.get('usage'),
        },
        'recorded_at': record.get('ts'),
    }


def dedupe_key(item: dict[str, Any]) -> str:
    """Same NPC(s) + same last user line + same offered tools = same test case."""
    messages = item['request']['messages']
    last_user = next((message_text(m) for m in reversed(messages) if m.get('role') == 'user'), '')
    tool_names = sorted(t['function']['name'] for t in item['request'].get('tools') or [] if 'function' in t)
    raw = json.dumps([item['tags'].get('npcs'), item['tags'].get('scenario'), last_user.strip().lower(), tool_names,
                      len(messages) > 2], ensure_ascii=False)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()


def load_transcript_turns(data_dir: str, since: str | None = None, conversation: str | None = None) -> list[dict[str, Any]]:
    turns = []
    for path in sorted(glob.glob(os.path.join(transcripts_dir(data_dir), '*.jsonl'))):
        for record in read_jsonl(path):
            if record.get('call_type', 'dialogue') != 'dialogue' or 'request' not in record:
                continue
            if conversation and record.get('conversation_id') != conversation:
                continue
            if since and (record.get('ts') or '') < since:
                continue
            turns.append(record)
    return turns


def select_turns(records: list[dict[str, Any]], include_followups: bool = False, include_images: bool = False,
                 include_errors: bool = False, scenarios: set[str] | None = None, limit: int | None = None,
                 seed: int = 7) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {'records': len(records), 'errors': 0, 'followups': 0, 'images': 0, 'scenario_filtered': 0, 'duplicates': 0}
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        response = record.get('response') or {}
        if not include_errors and (response.get('error') or (not response.get('content') and not response.get('tool_calls'))):
            stats['errors'] += 1
            continue
        item = turn_from_record(record)
        if not include_followups and item['tags']['followup_after_tool']:
            stats['followups'] += 1
            continue
        if not include_images and item['tags']['had_image']:
            stats['images'] += 1
            continue
        if scenarios and item['tags']['scenario'] not in scenarios:
            stats['scenario_filtered'] += 1
            continue
        key = dedupe_key(item)
        if key in seen:
            stats['duplicates'] += 1
            continue
        seen.add(key)
        items.append(item)
    if limit and len(items) > limit:
        items = stratified_sample(items, limit, seed)
    stats['selected'] = len(items)
    return items, stats


def stratified_sample(items: list[dict[str, Any]], limit: int, seed: int = 7) -> list[dict[str, Any]]:
    """Round-robin over (scenario, had_tool_call) groups so rare cases (radiant, tool turns) survive sampling."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault((item['tags']['scenario'], item['tags']['had_tool_call']), []).append(item)
    for group in groups.values():
        rng.shuffle(group)
    picked: list[dict[str, Any]] = []
    keys = sorted(groups, key=str)
    while len(picked) < limit and any(groups[k] for k in keys):
        for key in keys:
            if groups[key] and len(picked) < limit:
                picked.append(groups[key].pop())
    order = {id(item): i for i, item in enumerate(items)}
    return sorted(picked, key=lambda item: order[id(item)])
