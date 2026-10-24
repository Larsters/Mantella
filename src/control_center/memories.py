"""Read-only view of NPC memories for the Control Center's Memories tab.

Layout written by src/remember/summaries.py:
  <save_folder>/data/<Skyrim|Fallout4>/conversations/<world_id>/<Name - ref_id>/<Name>_summary_<N>.txt
  (legacy folders are just <Name>). Higher N = newer file (older files get condensed into a new one).
  Each memory is a paragraph, optionally preceded by an in-game time line like "[Day 1, 2 in the afternoon]".

Relationship records (W6, optional): relationship.json / relationships.json / <Name>_relationship.json in
the same NPC folder. Rendered as-is if present.

Files are parsed once per (mtime, size); polling is cheap.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Any

import src.usage_logger as usage_logger
import src.utils as utils
from src.control_center import usage

DAY_LINE = re.compile(r'^\[(Day [^\]]*)\]$')
SUMMARY_FILE = re.compile(r'^(?P<name>.+)_summary_(?P<n>\d+)\.txt$')
RELATIONSHIP_FILES = ('relationship.json', 'relationships.json')

_lock = threading.Lock()
_file_cache: dict[str, tuple[tuple[int, int], Any]] = {}


def conversations_root(config: Any) -> str:
    game = getattr(config, 'game', None)
    base_game = getattr(game, 'base_game', game)
    folder = 'Fallout4' if 'FALLOUT' in str(getattr(base_game, 'name', base_game)).upper() else 'Skyrim'
    return os.path.join(config.save_folder, 'data', folder, 'conversations')


def _split_folder(folder: str) -> tuple[str, str | None]:
    if ' - ' in folder:
        name, ref = folder.rsplit(' - ', 1)
        return name, ref
    return folder, None


def _stat_key(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def _cached(path: str, parse) -> Any:
    key = _stat_key(path)
    if key is None:
        return None
    with _lock:
        hit = _file_cache.get(path)
        if hit and hit[0] == key:
            return hit[1]
    value = parse(path)
    with _lock:
        _file_cache[path] = (key, value)
    return value


def parse_summary_file(path: str) -> list[dict[str, Any]]:
    """Paragraphs in file order, each with its optional [Day ...] prefix."""
    entries: list[dict[str, Any]] = []
    day: str | None = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = DAY_LINE.match(line)
            if m:
                day = m.group(1)
                continue
            entries.append({'day': day, 'text': line})
            day = None
    return entries


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {'_error': f'could not read {os.path.basename(path)}: {e}'}


def _safe_npc_dir(root: str, npc_id: str) -> str | None:
    """npc_id is '<world>/<folder>'; must stay inside root."""
    if not isinstance(npc_id, str) or npc_id.count('/') != 1:
        return None
    world, folder = npc_id.split('/')
    if world in ('', '.', '..') or folder in ('', '.', '..'):
        return None
    path = os.path.realpath(os.path.join(root, world, folder))
    root_real = os.path.realpath(root)
    if os.path.dirname(os.path.dirname(path)) != root_real or not os.path.isdir(path):
        return None
    return path


def _summary_files(npc_dir: str) -> list[tuple[int, str]]:
    files = []
    try:
        for entry in os.scandir(npc_dir):
            m = SUMMARY_FILE.match(entry.name)
            if m and entry.is_file():
                files.append((int(m.group('n')), entry.path))
    except OSError:
        pass
    return sorted(files)


def _relationship_path(npc_dir: str, name: str) -> str | None:
    for candidate in (*RELATIONSHIP_FILES, f'{name}_relationship.json', f'{name}_relationships.json'):
        path = os.path.join(npc_dir, candidate)
        if os.path.isfile(path):
            return path
    return None


def list_npcs(root: str, query: str | None = None) -> list[dict[str, Any]]:
    """Every NPC folder with at least one summary file, most recently updated first."""
    q = (query or '').strip().lower()
    result = []
    try:
        worlds = [w for w in os.scandir(root) if w.is_dir()]
    except OSError:
        return []
    for world in worlds:
        try:
            folders = [d for d in os.scandir(world.path) if d.is_dir()]
        except OSError:
            continue
        for d in folders:
            name, ref_id = _split_folder(d.name)
            if q and q not in name.lower() and q not in (ref_id or '').lower():
                continue
            files = _summary_files(d.path)
            if not files:
                continue
            mtime = max((os.stat(p).st_mtime for _, p in files), default=0)
            result.append({'id': f'{world.name}/{d.name}', 'name': name, 'ref_id': ref_id, 'world': world.name,
                           'updated': mtime, 'files': len(files)})
    result.sort(key=lambda r: -r['updated'])
    return result


def npc_memories(root: str, npc_id: str) -> dict[str, Any] | None:
    npc_dir = _safe_npc_dir(root, npc_id)
    if npc_dir is None:
        return None
    world, folder = npc_id.split('/')
    name, ref_id = _split_folder(folder)
    files = _summary_files(npc_dir)
    entries: list[dict[str, Any]] = []
    file_info = []
    updated = 0.0
    for number, path in files:
        parsed = _cached(path, parse_summary_file) or []
        mtime = os.stat(path).st_mtime
        updated = max(updated, mtime)
        file_info.append({'number': number, 'file': os.path.basename(path), 'entries': len(parsed), 'updated': mtime})
        for i, e in enumerate(parsed):
            digest = hashlib.sha1(f'{number}|{e["day"]}|{e["text"]}'.encode('utf-8')).hexdigest()[:12]
            entries.append({'id': digest, 'day': e['day'], 'text': e['text'], 'file_number': number,
                            'condensed': number < len(files) or None})
    entries.reverse()  # newest first: last file, last paragraph
    relationship = None
    rel_path = _relationship_path(npc_dir, name)
    if rel_path:
        relationship = _cached(rel_path, _read_json)
        updated = max(updated, os.stat(rel_path).st_mtime)
    return {'id': npc_id, 'name': name, 'ref_id': ref_id, 'world': world, 'updated': updated,
            'files': list(reversed(file_info)), 'entries': entries, 'relationship': relationship,
            'folder': npc_dir}


def find_npc_ids(root: str, names: list[str]) -> list[str]:
    """Match NPC names (as the game sends them, e.g. 'Whiterun Guard 2') to memory folders; newest world wins."""
    if not names:
        return []
    all_npcs = list_npcs(root)
    ids: list[str] = []
    for raw in names:
        base = utils.remove_trailing_number(raw).strip().lower()
        for npc in all_npcs:  # sorted newest first
            if npc['name'].lower() == base and npc['id'] not in ids:
                ids.append(npc['id'])
                break
    return ids


def active_conversation_names() -> tuple[list[str], str, str | None]:
    """NPC names of the current (or last) conversation and where they came from."""
    ctx = getattr(usage_logger, '_context', {}) or {}
    npcs = ctx.get('npcs')
    if isinstance(npcs, list) and npcs:
        return [n for n in npcs if isinstance(n, str)], 'conversation', ctx.get('conversation_id')
    try:
        for r in reversed(usage.load_records()):
            if r.get('call_type') == 'dialogue' and r.get('npcs'):
                return [n for n in r['npcs'] if isinstance(n, str)], 'last', r.get('conversation_id')
    except Exception:
        pass
    return [], 'recent', None


def active_payload(root: str) -> dict[str, Any]:
    names, source, conversation_id = active_conversation_names()
    ids = find_npc_ids(root, names)
    if not ids:
        source = 'recent' if not names else source
        if not names:
            ids = [n['id'] for n in list_npcs(root)[:3]]
    npcs = [m for m in (npc_memories(root, i) for i in ids) if m]
    missing = [n for n in names if utils.remove_trailing_number(n).strip().lower() not in {m['name'].lower() for m in npcs}]
    return {'source': source, 'conversation_id': conversation_id, 'names': names, 'npcs': npcs,
            'no_memories_yet': missing, 'root': root}
