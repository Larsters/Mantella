"""Per-NPC relationship record: what one NPC thinks of and knows about the player.

The record is a plain JSON-serialisable dict so that players can read and edit the file by hand.
All functions here are pure (no I/O, no LLM) to keep them easy to test.

Record layout (schema_version 1)::

    {
      "schema_version": 1,
      "npc_name": "Lydia", "npc_ref_id": "A2C94", "player_name": "Dragonborn",
      "opinion": -35,                      # -100 (hates) .. 100 (devoted)
      "opinion_label": "dislikes",         # derived from opinion, informational only
      "attitude": "Cold and curt since the insult.",
      "first_met_day": 12, "last_seen_day": 15.4, "conversation_count": 3,
      "facts":    [{"id": "f1", "text": "...", "day": 12}],
      "grudges":  [{"id": "g1", "text": "...", "day": 12, "severity": 2, "witnesses": ["Alfhild"]}],
      "favours":  [{"id": "d1", "text": "...", "owed_by": "player" | "npc", "day": 12}],
      "promises": [{"id": "p1", "text": "...", "by": "player" | "npc", "day": 12}],
      "history":  [{"day": 12, "change": -15, "reason": "..."}],
      "updated_at": "2026-10-24T18:00:00"
    }
"""
from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 1

OPINION_MIN = -100
OPINION_MAX = 100
MAX_OPINION_CHANGE_PER_CONVERSATION = 40

CAPS: dict[str, int] = {
    "facts": 12,
    "grudges": 8,
    "favours": 6,
    "promises": 6,
    "history": 8,
}
MAX_TEXT_LENGTH = 200
MAX_ATTITUDE_LENGTH = 160
MAX_WITNESSES = 6

ID_PREFIX: dict[str, str] = {"facts": "f", "grudges": "g", "favours": "d", "promises": "p"}

# (upper bound inclusive, label). Checked in order.
OPINION_LABELS: list[tuple[int, str]] = [
    (-60, "hates"),
    (-25, "dislikes"),
    (-8, "is wary of"),
    (7, "is neutral towards"),
    (24, "likes"),
    (59, "respects and trusts"),
    (OPINION_MAX, "is devoted to"),
]


def opinion_label(opinion: int) -> str:
    for bound, label in OPINION_LABELS:
        if opinion <= bound:
            return label
    return OPINION_LABELS[-1][1]


def clamp_opinion(value: Any) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(OPINION_MIN, min(OPINION_MAX, number))


def new_record(npc_name: str, npc_ref_id: str = "", player_name: str = "") -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "npc_name": npc_name,
        "npc_ref_id": str(npc_ref_id),
        "player_name": player_name,
        "opinion": 0,
        "opinion_label": opinion_label(0),
        "attitude": "",
        "first_met_day": None,
        "last_seen_day": None,
        "conversation_count": 0,
        "facts": [],
        "grudges": [],
        "favours": [],
        "promises": [],
        "history": [],
        "updated_at": None,
    }


def _clean_text(value: Any, max_length: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    value = " ".join(value.split())
    if len(value) > max_length:
        value = value[: max_length - 1].rstrip() + "…"
    return value


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def is_duplicate(a: str, b: str) -> bool:
    """Two entries are duplicates if their normalised text is equal, one contains the other,
    or their word sets overlap strongly (Jaccard >= 0.75)."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb or (len(na) > 12 and na in nb) or (len(nb) > 12 and nb in na):
        return True
    wa, wb = set(na.split()), set(nb.split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.75


def normalize_record(raw: Any, npc_name: str = "", npc_ref_id: str = "", player_name: str = "") -> dict[str, Any]:
    """Return a valid record from whatever was on disk (hand-edited, older version, partial)."""
    record = new_record(npc_name, npc_ref_id, player_name)
    if not isinstance(raw, dict):
        return record
    for key in ("npc_name", "npc_ref_id", "player_name"):
        if raw.get(key):
            record[key] = str(raw[key])
    record["opinion"] = clamp_opinion(raw.get("opinion", 0))
    record["opinion_label"] = opinion_label(record["opinion"])
    record["attitude"] = _clean_text(raw.get("attitude", ""), MAX_ATTITUDE_LENGTH)
    for key in ("first_met_day", "last_seen_day"):
        value = raw.get(key)
        record[key] = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    try:
        record["conversation_count"] = max(0, int(raw.get("conversation_count", 0)))
    except (TypeError, ValueError):
        record["conversation_count"] = 0
    record["updated_at"] = raw.get("updated_at")

    for key in ("facts", "grudges", "favours", "promises"):
        items = raw.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                continue
            _add_item(record, key, item, default_day=item.get("day"))
    history = raw.get("history") or []
    if isinstance(history, list):
        record["history"] = [h for h in history if isinstance(h, dict)][-CAPS["history"]:]
    _enforce_caps(record)
    return record


def _next_id(record: dict[str, Any], key: str) -> str:
    prefix = ID_PREFIX[key]
    used = {item.get("id") for item in record[key]}
    n = 1
    while f"{prefix}{n}" in used:
        n += 1
    return f"{prefix}{n}"


def _sanitize_item(key: str, item: dict[str, Any], default_day: Any) -> dict[str, Any] | None:
    text = _clean_text(item.get("text", ""))
    if not text:
        return None
    day = item.get("day", default_day)
    if not isinstance(day, (int, float)) or isinstance(day, bool):
        day = default_day if isinstance(default_day, (int, float)) else None
    if isinstance(day, float):
        day = int(day)
    clean: dict[str, Any] = {"text": text, "day": day}
    if key == "grudges":
        try:
            severity = int(item.get("severity", 1))
        except (TypeError, ValueError):
            severity = 1
        clean["severity"] = max(1, min(3, severity))
        witnesses = item.get("witnesses") or []
        if isinstance(witnesses, str):
            witnesses = [w.strip() for w in witnesses.split(",")]
        clean["witnesses"] = [_clean_text(w, 60) for w in witnesses if isinstance(w, str) and w.strip()][:MAX_WITNESSES]
    elif key == "favours":
        owed_by = str(item.get("owed_by", "player")).lower()
        clean["owed_by"] = "npc" if owed_by in ("npc", "them", "character") else "player"
    elif key == "promises":
        by = str(item.get("by", "player")).lower()
        clean["by"] = "npc" if by in ("npc", "them", "character") else "player"
    return clean


def _add_item(record: dict[str, Any], key: str, item: dict[str, Any], default_day: Any) -> None:
    clean = _sanitize_item(key, item, default_day)
    if clean is None:
        return
    for existing in record[key]:
        if is_duplicate(existing["text"], clean["text"]):
            # Merge into the existing entry: keep the newer wording/day, union witnesses, max severity.
            existing["text"] = clean["text"]
            if clean.get("day") is not None:
                existing["day"] = clean["day"]
            if key == "grudges":
                existing["severity"] = max(existing.get("severity", 1), clean["severity"])
                witnesses = existing.get("witnesses", [])
                for w in clean["witnesses"]:
                    if w not in witnesses:
                        witnesses.append(w)
                existing["witnesses"] = witnesses[:MAX_WITNESSES]
            return
    given_id = item.get("id")
    if isinstance(given_id, str) and given_id and given_id not in {i.get("id") for i in record[key]}:
        clean = {"id": given_id, **clean}
    else:
        clean = {"id": _next_id(record, key), **clean}
    record[key].append(clean)


def _remove_items(record: dict[str, Any], key: str, refs: Any) -> list[dict[str, Any]]:
    """Remove items referenced by id or by (approximate) text. Returns the removed items."""
    if not isinstance(refs, list):
        refs = [refs] if refs else []
    removed = []
    for ref in refs:
        if isinstance(ref, dict):
            ref = ref.get("id") or ref.get("text") or ""
        ref = str(ref).strip()
        if not ref:
            continue
        for item in list(record[key]):
            if item.get("id") == ref or is_duplicate(item["text"], ref):
                record[key].remove(item)
                removed.append(item)
                break
    return removed


def _day_key(item: dict[str, Any]) -> float:
    day = item.get("day")
    return day if isinstance(day, (int, float)) else -1


def _enforce_caps(record: dict[str, Any]) -> None:
    for key in ("facts", "favours", "promises"):
        cap = CAPS[key]
        if len(record[key]) > cap:
            # Keep the most recent entries (stable for equal days: later entries win)
            indexed = sorted(enumerate(record[key]), key=lambda p: (_day_key(p[1]), p[0]))
            keep = {i for i, _ in indexed[-cap:]}
            record[key] = [item for i, item in enumerate(record[key]) if i in keep]
    cap = CAPS["grudges"]
    if len(record["grudges"]) > cap:
        # Drop the least severe, oldest grudges first
        indexed = sorted(enumerate(record["grudges"]), key=lambda p: (p[1].get("severity", 1), _day_key(p[1]), p[0]))
        keep = {i for i, _ in indexed[-cap:]}
        record["grudges"] = [item for i, item in enumerate(record["grudges"]) if i in keep]
    record["history"] = record["history"][-CAPS["history"]:]


def decayed_opinion(record: dict[str, Any], current_day: float | None, decay_per_day: float) -> int:
    """Opinion drifts towards neutral by `decay_per_day` points per in-game day since last seen.
    Grudges keep a floor: an NPC holding a grudge of severity s never drifts above -8*s."""
    opinion = clamp_opinion(record.get("opinion", 0))
    last_seen = record.get("last_seen_day")
    if decay_per_day <= 0 or current_day is None or not isinstance(last_seen, (int, float)):
        return opinion
    days = max(0.0, float(current_day) - float(last_seen))
    drift = int(days * decay_per_day)
    if drift <= 0 or opinion == 0:
        return opinion
    if opinion > 0:
        return max(0, opinion - drift)
    decayed = min(0, opinion + drift)
    floor = -8 * max((g.get("severity", 1) for g in record.get("grudges", [])), default=0)
    if floor < 0 and opinion <= floor:
        decayed = min(decayed, floor)
    return decayed


def apply_update(record: dict[str, Any], update: dict[str, Any], game_day: float | None,
                 decay_per_day: float = 0.0, now: datetime | None = None) -> dict[str, Any]:
    """Merge an LLM-produced update (delta) into a copy of `record` and return it.

    Update format (all keys optional)::

        {"opinion_change": -15, "reason": "...", "attitude": "...",
         "new_facts": ["..."], "new_grudges": [{"text": "...", "severity": 2, "witnesses": ["..."]}],
         "resolved_grudges": ["g1"], "new_favours": [{"text": "...", "owed_by": "player"}],
         "settled_favours": ["d1"], "new_promises": [{"text": "...", "by": "player"}],
         "resolved_promises": [{"id": "p1", "status": "kept"}]}
    """
    result = copy.deepcopy(record)
    day = int(game_day) if isinstance(game_day, (int, float)) and not isinstance(game_day, bool) else None

    base_opinion = decayed_opinion(result, game_day, decay_per_day)
    try:
        change = int(round(float(update.get("opinion_change", 0) or 0)))
    except (TypeError, ValueError):
        change = 0
    change = max(-MAX_OPINION_CHANGE_PER_CONVERSATION, min(MAX_OPINION_CHANGE_PER_CONVERSATION, change))
    result["opinion"] = clamp_opinion(base_opinion + change)
    result["opinion_label"] = opinion_label(result["opinion"])
    reason = _clean_text(update.get("reason", ""), MAX_TEXT_LENGTH)
    if change != 0 or reason:
        result["history"].append({"day": day, "change": change, "reason": reason})

    attitude = _clean_text(update.get("attitude", ""), MAX_ATTITUDE_LENGTH)
    if attitude:
        result["attitude"] = attitude

    for item in update.get("new_facts") or []:
        _add_item(result, "facts", item if isinstance(item, dict) else {"text": item}, day)
    for item in update.get("new_grudges") or []:
        _add_item(result, "grudges", item if isinstance(item, dict) else {"text": item}, day)
    for item in update.get("new_favours") or []:
        _add_item(result, "favours", item if isinstance(item, dict) else {"text": item}, day)
    for item in update.get("new_promises") or []:
        _add_item(result, "promises", item if isinstance(item, dict) else {"text": item}, day)

    for grudge in _remove_items(result, "grudges", update.get("resolved_grudges")):
        result["history"].append({"day": day, "change": 0, "reason": f"Forgave: {grudge['text']}"})
    _remove_items(result, "favours", update.get("settled_favours"))
    for entry in update.get("resolved_promises") or []:
        status = entry.get("status", "resolved") if isinstance(entry, dict) else "resolved"
        for promise in _remove_items(result, "promises", [entry]):
            if status in ("kept", "broken"):
                result["history"].append({"day": day, "change": 0, "reason": f"Promise {status}: {promise['text']}"})

    if day is not None:
        if result.get("first_met_day") is None:
            result["first_met_day"] = day
        result["last_seen_day"] = round(float(game_day), 2)
    result["conversation_count"] = int(result.get("conversation_count", 0)) + 1
    result["updated_at"] = (now or datetime.now()).isoformat(timespec="seconds")
    result["schema_version"] = SCHEMA_VERSION
    _enforce_caps(result)
    return result
