"""Trust and honesty (W12): NPCs comply with requests in proportion to trust, verify checkable claims
with their tools before acting on them, and remember being lied to.

Everything is gated by `trust_and_honesty_enabled` (default False = upstream behaviour). Hooks:
- `apply_prompt_rules(config)`        called after the prompt profile is applied (config load)
- `function_prompt(config, base)`     FunctionClient: static rules appended to `function_llm_prompt`
- `publish_scene(...)` / `scene_text()` Context publishes who is talking and their relationship text;
                                      FunctionClient appends it to the tool model's system prompt
- `tool_description(action, config)`  FunctionManager: `trust_description` from data/actions/*.json
- `track_lies(config)`                relationships: lies + credibility in the end-of-conversation update
"""
from __future__ import annotations

import threading
from typing import Any

from src.trust import rules

MAX_BIO_CHARS = 240

_SINGLE_MARKER = "\n\nCharacter\n"
_MULTI_MARKER = "\n\nCharacters\n"


def is_enabled(config: Any) -> bool:
    return bool(getattr(config, "trust_and_honesty_enabled", False))


def track_lies(config: Any) -> bool:
    return is_enabled(config)


# ---------- main NPC prompts ----------
def insert_block(prompt: str, block: str, marker: str) -> str:
    """Put `block` before `marker` (the Character section of living_npcs_v1); if the prompt has no such
    section (custom prompt), put it in front. Idempotent."""
    if not prompt or block in prompt:
        return prompt
    if marker in prompt:
        return prompt.replace(marker, f"\n\n{block}{marker}", 1)
    return f"{block}\n\n{prompt}"


def apply_prompt_rules(config: Any) -> None:
    """Adds the rule blocks to the one-on-one and multi-NPC prompts (not radiant: no player there)."""
    if not is_enabled(config):
        return
    if getattr(config, "prompt", None):
        config.prompt = insert_block(config.prompt, rules.SINGLE_BLOCK, _SINGLE_MARKER)
    if getattr(config, "multi_npc_prompt", None):
        config.multi_npc_prompt = insert_block(config.multi_npc_prompt, rules.MULTI_BLOCK, _MULTI_MARKER)


# ---------- tool-calling model ----------
def function_prompt(config: Any, base: str) -> str:
    if not is_enabled(config) or rules.FUNCTION_BLOCK in base:
        return base
    return f"{base.rstrip()}\n\n{rules.FUNCTION_BLOCK}"


_scene_lock = threading.Lock()
_scene: dict[str, str] = {"text": ""}


def build_scene(npcs: list[Any], player_name: str, trusts: str, relationship: str) -> str:
    """Short, volatile description of who is talking, for the tool model (it never sees the NPC prompt)."""
    lines = ["Scene"]
    for npc in npcs:
        bio = " ".join(str(getattr(npc, "bio", "") or "").split())
        if len(bio) > MAX_BIO_CHARS:
            bio = bio[:MAX_BIO_CHARS].rsplit(" ", 1)[0] + "…"
        lines.append(f"- {npc.name}: {bio}" if bio else f"- {npc.name}")
    player = player_name or "The player"
    if trusts:
        lines.append(f"{player} is {trusts}.")
    if relationship:
        lines.append(relationship.strip())
    return "\n".join(lines) if len(lines) > 1 else ""


def publish_scene(config: Any, npcs: list[Any], player_name: str, trusts: str, relationship: str) -> None:
    """Called by Context whenever it builds the NPC system prompt. One conversation runs at a time."""
    if not (is_enabled(config) and getattr(config, "trust_tool_model_context", True)):
        return
    try:
        text = build_scene([n for n in npcs if not getattr(n, "is_player_character", False)], player_name, trusts, relationship)
    except Exception:
        text = ""
    with _scene_lock:
        _scene["text"] = text


def clear_scene() -> None:
    with _scene_lock:
        _scene["text"] = ""


def scene_text() -> str:
    with _scene_lock:
        return _scene["text"]


def function_system_prompt(config: Any, static_prompt: str) -> str:
    """Static function prompt + the current scene (last, so the static part stays cacheable)."""
    if not (is_enabled(config) and getattr(config, "trust_tool_model_context", True)):
        return static_prompt
    scene = scene_text()
    return f"{static_prompt}\n\n{scene}" if scene else static_prompt


# ---------- tool descriptions ----------
def tool_description(action: dict[str, Any], config: Any) -> str:
    """Description sent for a tool: trust override > guided description > upstream description."""
    if is_enabled(config):
        override = action.get("trust_description")  # data/actions/*.json
        if override:
            return override
    if getattr(config, "guided_action_descriptions", False) and action.get("guided_description"):
        return action["guided_description"]
    return action["description"]
