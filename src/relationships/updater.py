"""One LLM call per conversation (per group of NPCs who heard the same messages) that turns the
transcript plus the current records into structured relationship updates.

Why a separate call instead of piggybacking on the summary call: the memory prompt is free text that
users edit and its output is appended verbatim to the summary .txt files. Asking the same call for
JSON would change upstream's summary format and break user prompts. A separate, small call keeps the
summaries untouched and can be switched off on its own. It uses the summary client (cheap model) and
runs in a background thread so ending a conversation is not slowed down.
"""
from __future__ import annotations

import copy
import json
import threading
from typing import Any

from src import utils
from src.relationships.parser import parse_update_reply

logger = utils.get_logger()

DEFAULT_UPDATE_PROMPT = """You maintain long-term relationship memory for characters in {game}. You will read a conversation between {player_name} (the player) and {names}, together with each character's current relationship record, and output ONLY the changes this conversation causes, as JSON.

Rules:
- Judge from each character's own perspective, personality and values (see their bio). A proud warrior takes insults harder than a drunk; a beggar is grateful for a few coins.
- opinion_change: an integer from -40 to 40. Use 0 for small talk, +-3 to 10 for mild, +-10 to 25 for notable, +-25 to 40 for severe events (assault, betrayal, saving their life).
- Only record what happened in THIS conversation and is not already in the record. Refer to existing record items by their id.
- new_facts: lasting facts the character learned about {player_name} (name, origin, deeds, plans, beliefs, preferences). No trivia. At most 3.
- new_grudges: insults, threats, harm, humiliation or broken promises against the character or people they care about. Describe the concrete incident from the character's point of view, e.g. "{player_name} called me a coward in front of Alfhild". witnesses: names of other people who saw it (from the participants or nearby list). severity: 1 minor, 2 serious, 3 unforgivable.
- resolved_grudges: ids of grudges that {player_name} sincerely apologised for or made up for.
- new_favours: owed_by "player" if {player_name} owes the character, "npc" if the character owes {player_name}. settled_favours: ids of favours that were repaid.
- new_promises: explicit commitments only; by "player" or "npc". resolved_promises: objects with the promise "id" and "status" ("kept" or "broken").
- attitude: one short sentence (max 20 words) describing how the character now feels about {player_name}, in the third person.
- reason: a few words explaining the opinion change.
- Return one entry in "npcs" for every character listed under "Characters to update", using the exact name. Use empty lists when nothing changed.
- Write all text in {language}.

Return only the JSON object."""

_STR_LIST = {"type": "array", "items": {"type": "string"}}

UPDATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["npcs"],
    "properties": {
        "npcs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "opinion_change", "reason", "attitude", "new_facts", "new_grudges",
                             "resolved_grudges", "new_favours", "settled_favours", "new_promises", "resolved_promises"],
                "properties": {
                    "name": {"type": "string"},
                    "opinion_change": {"type": "integer"},
                    "reason": {"type": "string"},
                    "attitude": {"type": "string"},
                    "new_facts": _STR_LIST,
                    "new_grudges": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["text", "severity", "witnesses"],
                        "properties": {"text": {"type": "string"}, "severity": {"type": "integer"}, "witnesses": _STR_LIST}}},
                    "resolved_grudges": _STR_LIST,
                    "new_favours": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False, "required": ["text", "owed_by"],
                        "properties": {"text": {"type": "string"}, "owed_by": {"type": "string", "enum": ["player", "npc"]}}}},
                    "settled_favours": _STR_LIST,
                    "new_promises": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False, "required": ["text", "by"],
                        "properties": {"text": {"type": "string"}, "by": {"type": "string", "enum": ["player", "npc"]}}}},
                    "resolved_promises": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False, "required": ["id", "status"],
                        "properties": {"id": {"type": "string"}, "status": {"type": "string", "enum": ["kept", "broken"]}}}},
                },
            },
        }
    },
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "relationship_update", "strict": True, "schema": UPDATE_JSON_SCHEMA},
}

MIN_OUTPUT_TOKENS = 800
MAX_BIO_CHARS = 600


def record_for_prompt(record: dict[str, Any]) -> str:
    """Compact JSON of the parts of a record the LLM needs (ids included, history excluded)."""
    view = {
        "opinion": record.get("opinion", 0),
        "attitude": record.get("attitude", ""),
        "facts": [{"id": f["id"], "text": f["text"]} for f in record.get("facts", [])],
        "grudges": [{"id": g["id"], "text": g["text"], "severity": g.get("severity", 1)} for g in record.get("grudges", [])],
        "favours": [{"id": d["id"], "text": d["text"], "owed_by": d.get("owed_by", "player")} for d in record.get("favours", [])],
        "promises": [{"id": p["id"], "text": p["text"], "by": p.get("by", "player")} for p in record.get("promises", [])],
    }
    return json.dumps(view, ensure_ascii=False)


BYSTANDER_INSTRUCTIONS = """Bystanders ({bystanders}) were nearby and only overheard the conversation; they did not take part. Also return one entry in "npcs" for each of them, using the exact name. For a bystander: opinion_change from -15 to 15 and only if what they overheard would plausibly change how they see {player_name} (0 for small talk); new_facts only what they overheard; new_grudges only for things that hurt them or people they care about (severity 1 or 2); leave favours and promises empty."""


def _npc_section(name: str, bio: str, record: dict[str, Any], heading_suffix: str = "") -> str:
    bio_text = (bio or "").strip().replace("\n", " ")
    if len(bio_text) > MAX_BIO_CHARS:
        bio_text = bio_text[:MAX_BIO_CHARS] + "…"
    return f"\n## {name}{heading_suffix}\nBio: {bio_text or 'unknown'}\nCurrent record: {record_for_prompt(record)}"


def build_messages(prompt_template: str, game: str, language: str, player_name: str,
                   npcs: list[tuple[str, str, dict[str, Any]]], witnesses: list[str],
                   transcript: str, game_day: float | None,
                   bystanders: list[tuple[str, str, dict[str, Any]]] | None = None) -> tuple[str, str]:
    """Returns (system_prompt, user_content). `npcs` and `bystanders` are lists of (name, bio, record).
    Bystanders (nearby NPCs who only overheard) are updated in the same call; their instructions go into the
    user message so that custom update prompts keep working."""
    bystanders = bystanders or []
    names = ", ".join(name for name, _, _ in npcs)
    system_prompt = prompt_template.format(game=game, language=language, player_name=player_name, names=names)
    lines = [f"Characters to update: {names}"]
    if bystanders:
        bystander_names = ", ".join(name for name, _, _ in bystanders)
        lines.append(f"Bystanders to update (overheard only): {bystander_names}")
        lines.append(BYSTANDER_INSTRUCTIONS.format(bystanders=bystander_names, player_name=player_name))
    if witnesses:
        lines.append(f"Other people present or nearby (possible witnesses): {', '.join(witnesses)}")
    if isinstance(game_day, (int, float)):
        lines.append(f"Current in-game day: {int(game_day)}")
    for name, bio, record in npcs:
        lines.append(_npc_section(name, bio, record))
    for name, bio, record in bystanders:
        lines.append(_npc_section(name, bio, record, " (bystander, overheard only)"))
    lines.append(f"\n## Conversation transcript\n{transcript.strip()}")
    return system_prompt, "\n".join(lines)


def _request_params_for(client: Any, structured: bool) -> dict[str, Any]:
    params = dict(getattr(client, "_request_params", None) or {})
    params.pop("stop", None)    # upstream summary params stop at "#", JSON must not be cut
    params.pop("stream", None)
    params.pop("tools", None)
    params.pop("tool_choice", None)
    token_key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
    try:
        current = int(params.get(token_key) or 0)
    except (TypeError, ValueError):
        current = 0
    params[token_key] = max(current, MIN_OUTPUT_TOKENS)
    if structured:
        params["response_format"] = RESPONSE_FORMAT
    else:
        params.pop("response_format", None)
    return params


def _call(client: Any, messages: Any, structured: bool) -> str | None:
    # Shallow copy so the shared client's params are not mutated; the copy still goes through
    # ClientBase._request_call_full (and with it any request adapters / usage logging).
    call_client = copy.copy(client)
    call_client._request_params = _request_params_for(client, structured)
    # Own lock: this runs in a background thread and must not block the NPC's next streamed reply
    # when the summary client is the main conversation client.
    call_client._generation_lock = threading.Lock()
    completion = call_client._request_call_full(messages)
    if not completion or not getattr(completion, "choices", None):
        return None
    return completion.choices[0].message.content


def request_update(client: Any, messages: Any) -> dict[str, Any] | None:
    """Ask for structured output first; on an API error or unparseable reply, retry once as plain text."""
    reply = None
    try:
        reply = _call(client, messages, structured=True)
    except Exception as e:  # providers / local servers that reject response_format
        logger.info(f"Relationship update: structured output not accepted ({e.__class__.__name__}); retrying as plain JSON.")
    parsed = parse_update_reply(reply)
    if parsed is not None:
        return parsed
    try:
        reply = _call(client, messages, structured=False)
    except Exception as e:
        logger.error(f"Relationship update failed: {e}")
        return None
    parsed = parse_update_reply(reply)
    if parsed is None:
        logger.warning(f"Relationship update: could not parse LLM reply: {str(reply)[:300]}")
    return parsed
