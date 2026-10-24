"""Turns relationship records into short prompt text and adjusts the {trust} wording."""
from __future__ import annotations

import math
from typing import Any, Callable

from src.relationships.record import CREDIBILITY_DEFAULT, decayed_opinion, has_met, opinion_label

TokenCounter = Callable[[str], int]


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token for English)."""
    return math.ceil(len(text) / 4)


def _t(text: str) -> str:
    return str(text).strip().rstrip(" .!;")


def _when(day: Any, current_day: float | None) -> str:
    if not isinstance(day, (int, float)):
        return ""
    if isinstance(current_day, (int, float)):
        ago = int(current_day) - int(day)
        if ago <= 0:
            return " (today)"
        if ago == 1:
            return " (yesterday)"
        if ago < 30:
            return f" ({ago} days ago)"
    return f" (Day {int(day)})"


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")


def credibility_text(record: dict[str, Any], name: str, player: str) -> str:
    """One sentence on how far the NPC takes the player's word ("" for the neutral default)."""
    try:
        credibility = int(record.get("credibility", CREDIBILITY_DEFAULT))
    except (TypeError, ValueError):
        credibility = CREDIBILITY_DEFAULT
    if credibility <= 20:
        return f"{name} believes nothing {player} claims without proof and refuses favours that rest on their word."
    if credibility <= 35:
        return f"{name} doubts {player}'s word and checks their claims before acting on them."
    if credibility >= 70:
        return f"{name} has found {player} honest and takes their word."
    return ""


def lies_text(record: dict[str, Any], name: str, player: str, current_day: float | None) -> str:
    lies = record.get("lies") or []
    if not lies:
        return ""
    latest = max(lies, key=lambda l: (l.get("day") or 0))
    note = _when(latest.get("day"), current_day)
    if latest.get("overheard"):
        note = f"{note[:-1]}, overheard)" if note else " (overheard)"
    return f"{name} has caught {player} lying {_times(len(lies))}, most recently: {_t(latest['text'])}{note}."


def render_record(record: dict[str, Any], player_name: str, current_day: float | None,
                  token_budget: int, decay_per_day: float = 0.0,
                  count_tokens: TokenCounter = approx_tokens) -> str:
    """Render one NPC's record as a few short sentences, most important first, within the budget.

    Priority: opinion/attitude > grudges (by severity) > open promises > debts > facts (newest first).
    """
    name = record.get("npc_name") or "This character"
    player = player_name or record.get("player_name") or "the player"
    has_content = any(record.get(k) for k in ("facts", "grudges", "favours", "promises", "attitude", "lies")) \
        or has_met(record)
    if not has_content:
        return ""

    opinion = decayed_opinion(record, current_day, decay_per_day)
    head = f"{name} {opinion_label(opinion)} {player} (opinion {opinion:+d} of 100)."
    if record.get("attitude"):
        head += f" {record['attitude']}"
    credibility = credibility_text(record, name, player)  # trust and honesty: always kept with the opinion
    if credibility:
        head += f" {credibility}"

    pieces: list[str] = []
    lied = lies_text(record, name, player, current_day)
    if lied:
        pieces.append(lied)
    grudges = sorted(record.get("grudges", []), key=lambda g: (-g.get("severity", 1), -(g.get("day") or 0)))
    for g in grudges:
        notes = []
        when = _when(g.get("day"), current_day)
        if when:
            notes.append(when.strip(" ()"))
        if g.get("overheard"):
            notes.append("overheard")
        if g.get("witnesses"):
            notes.append("witnessed by " + ", ".join(g["witnesses"]))
        suffix = f" ({', '.join(notes)})" if notes else ""
        pieces.append(f"Grudge: {_t(g['text'])}{suffix}.")
    for p in record.get("promises", []):
        who = player if p.get("by") == "player" else name
        pieces.append(f"Open promise by {who}: {_t(p['text'])}.")
    for d in record.get("favours", []):
        if d.get("owed_by") == "player":
            pieces.append(f"{player} owes {name}: {_t(d['text'])}.")
        else:
            pieces.append(f"{name} owes {player}: {_t(d['text'])}.")
    facts = sorted(record.get("facts", []), key=lambda f: -(f.get("day") or 0))
    if facts:
        pieces.append("__FACTS__")

    text = head
    if count_tokens(text) > token_budget:
        return text  # the opinion line is always kept
    for piece in pieces:
        if piece == "__FACTS__":
            known: list[str] = []
            for f in facts:
                fact = _t(f["text"]) + (" (overheard)" if f.get("overheard") else "")
                trial = f"{name} knows about {player}: " + "; ".join(known + [fact]) + "."
                if count_tokens(text + " " + trial) > token_budget:
                    break
                known.append(fact)
            if known:
                text += " " + f"{name} knows about {player}: " + "; ".join(known) + "."
            continue
        if count_tokens(text + " " + piece) > token_budget:
            continue  # try the next (possibly shorter) piece
        text += " " + piece
    return text


def render_records(records: list[dict[str, Any]], player_name: str, current_day: float | None,
                   token_budget: int, decay_per_day: float = 0.0,
                   count_tokens: TokenCounter = approx_tokens) -> str:
    """Render several NPCs' records, splitting the token budget between them."""
    non_empty = [r for r in records if r]
    if not non_empty:
        return ""
    per_npc = max(40, token_budget // len(non_empty))
    rendered = [render_record(r, player_name, current_day, per_npc, decay_per_day, count_tokens) for r in non_empty]
    rendered = [r for r in rendered if r]
    if not rendered:
        return ""
    return "Relationship with the player:\n" + "\n".join(rendered)


# Ordered from most negative to most positive; None = no adjective
_TRUST_ADJECTIVES: list[tuple[int, str | None]] = [
    (-60, "despised"),
    (-25, "disliked"),
    (-8, "distrusted"),
    (7, None),
    (24, "well-liked"),
    (59, "valued"),
    (100, "deeply trusted"),
]


def adjust_trust(base_trust: str, opinion: int | None) -> str:
    """Combine upstream's familiarity wording ("a stranger", "a friend") with the NPC's opinion.

    e.g. ("an acquaintance", -40) -> "a disliked acquaintance". Enemies stay enemies.
    """
    if opinion is None or not base_trust:
        return base_trust
    if base_trust.strip() == "an enemy":
        return base_trust
    adjective = None
    for bound, adj in _TRUST_ADJECTIVES:
        if opinion <= bound:
            adjective = adj
            break
    if not adjective:
        return base_trust
    words = base_trust.split(" ", 1)
    noun = words[1] if len(words) == 2 and words[0] in ("a", "an") else base_trust
    return f"a {adjective} {noun}"
