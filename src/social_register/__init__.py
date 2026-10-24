"""Social register: low-class / street NPCs (beggars, thieves, dock hands, skooma users...) and all Khajiit
talk in street cant. Rendered through the existing `{speech_flavor}` prompt slot, appended after the
speech-flavour (German mix) text, so no prompt edits are needed.

Public API:
- compose_speech_flavor(config, characters, flavor_text) -> str   flavor_text + street register text
- get_social_register_text(config, characters, with_tail=True) -> str
- detect_register(name, bio, voice_type, race) -> "street" | "khajiit" | None

Config: `social_register_enabled` (default False = upstream behaviour).
Per-NPC override in `<save folder>/data/Skyrim/character_overrides/*.json`:
`"social_register": "street" | "khajiit" | "none"` ("auto" = detection). An override that forces a register
works even while the global setting is off; "none" disables it for that NPC.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Iterable

from src.social_register.data import (
    BIO_KEYWORDS, EXCLUDED_NAMES, EXCLUDED_NAME_FRAGMENTS, EXCLUDED_VOICES, KHAJIIT_TEXT, KHAJIIT_VOICES,
    STREET_NAMES, STREET_TEXT, STREET_VOICES, TAIL,
)

STREET = "street"
KHAJIIT = "khajiit"
_OFF_VALUES = {"none", "off", "no", "false", "disabled", "0"}
_AUTO_VALUES = {"", "auto", "default"}
_BIO_OPENING_CHARS = 300
_OVERRIDE_KEY = "social_register"


@lru_cache(maxsize=None)
def _keyword_regex(keyword: str) -> re.Pattern:
    return re.compile(r"(?<![\w'])" + re.escape(keyword) + r"(?:s|es)?(?![\w'])", re.IGNORECASE)


def _is_khajiit(race: str | None, voice_type: str) -> bool:
    flat = re.sub(r"[^a-z]", "", str(race or "").lower())
    return "khajiit" in flat or voice_type in KHAJIIT_VOICES


def detect_register(name: str, bio: str = "", voice_type: str = "", race: str | None = None) -> str | None:
    """Street register from what the server knows. Khajiit always (their own variant); guards and Lydia never."""
    lname = str(name or "").strip().lower()
    voice = str(voice_type or "").strip().lower()
    if lname in EXCLUDED_NAMES or any(f in lname for f in EXCLUDED_NAME_FRAGMENTS) or voice in EXCLUDED_VOICES:
        return None
    if _is_khajiit(race, voice):
        return KHAJIIT
    if lname in STREET_NAMES or voice in STREET_VOICES:
        return STREET
    opening = str(bio or "")[:_BIO_OPENING_CHARS]
    if any(_keyword_regex(k).search(opening) for k in BIO_KEYWORDS):
        return STREET
    return None


# ---------------------------------------------------------------- overrides
_override_cache: dict[str, tuple[float, dict[str, list[tuple[str, str]]]]] = {}


def _overrides_folder(config: Any) -> str | None:
    save_folder = getattr(config, "save_folder", None)
    if not save_folder:
        return None
    game_folder = "Skyrim"
    try:
        if "FALLOUT" in config.game.base_game.name:
            game_folder = "Fallout4"
    except Exception:
        pass
    return os.path.join(save_folder, "data", game_folder, "character_overrides")


def load_overrides(folder: str | None) -> dict[str, list[tuple[str, str]]]:
    """{lower name: [(base_id, value)]} from override JSONs that set `social_register`; cached by mtime."""
    if not folder or not os.path.isdir(folder):
        return {}
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")]
        stamp = max([os.path.getmtime(folder)] + [os.path.getmtime(f) for f in files])
    except OSError:
        return {}
    cached = _override_cache.get(folder)
    if cached and cached[0] == stamp:
        return cached[1]
    result: dict[str, list[tuple[str, str]]] = {}
    for path in files:
        try:
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue
        for entry in data if isinstance(data, list) else [data]:
            if not isinstance(entry, dict) or _OVERRIDE_KEY not in entry or not entry.get("name"):
                continue
            result.setdefault(str(entry["name"]).strip().lower(), []).append(
                (str(entry.get("base_id", "") or ""), str(entry[_OVERRIDE_KEY])))
    _override_cache[folder] = (stamp, result)
    return result


def _norm_id(value: str) -> str:
    return str(value or "").strip().lower().removeprefix("0x").lstrip("0")


def find_override(overrides: dict[str, list[tuple[str, str]]], name: str, base_id: str = "") -> str | None:
    fallback = None
    for entry_base_id, value in overrides.get(str(name or "").strip().lower(), []):
        if entry_base_id and _norm_id(entry_base_id) == _norm_id(base_id):
            return value
        if not entry_base_id:
            fallback = value
    return fallback


def resolve(enabled: bool, character: Any, overrides: dict[str, list[tuple[str, str]]] | None = None) -> str | None:
    """Effective register for one NPC: "street", "khajiit" or None."""
    name = getattr(character, "name", "")
    race = getattr(character, "race", None)
    voice = str(getattr(character, "in_game_voice_model", "") or "")
    override = find_override(overrides or {}, name, str(getattr(character, "base_id", "") or ""))
    if override is not None:
        v = override.strip().lower()
        if v in _OFF_VALUES:
            return None
        if v in (STREET, "on", "true", "yes", "low", "low-class"):
            return KHAJIIT if _is_khajiit(race, voice.lower()) else STREET
        if v == KHAJIIT:
            return KHAJIIT
        # "auto" / unknown values fall through to detection, which runs only while enabled
    if not enabled:
        return None
    return detect_register(name, getattr(character, "bio", "") or "", voice, race)


# ---------------------------------------------------------------- rendering
def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _render(template: str, names: list[str]) -> str:
    plural = len(names) > 1
    return template.format(names=_join(names), s="" if plural else "s", self="themselves" if plural else "themself")


def build_text(npcs: list[tuple[str, str | None]], with_tail: bool = True) -> str:
    """npcs: [(name, register)] -> text; "" when nobody has a register."""
    street = [n for n, r in npcs if r == STREET]
    khajiit = [n for n, r in npcs if r == KHAJIIT]
    parts = []
    if street:
        parts.append(_render(STREET_TEXT, street))
    if khajiit:
        parts.append(_render(KHAJIIT_TEXT, khajiit))
    if not parts:
        return ""
    if with_tail:
        parts.append(TAIL)
    return " ".join(parts)


def get_social_register_text(config: Any, characters: Iterable[Any], with_tail: bool = True) -> str:
    enabled = bool(getattr(config, "social_register_enabled", False))
    overrides = load_overrides(_overrides_folder(config))
    if not enabled and not overrides:
        return ""
    npcs = [(c.name, resolve(enabled, c, overrides)) for c in characters if not getattr(c, "is_player_character", False)]
    return build_text(npcs, with_tail)


def compose_speech_flavor(config: Any, characters: Iterable[Any], flavor_text: str) -> str:
    """Hook for context.py: the `{speech_flavor}` value = speech flavour text + street register text.
    The speech flavour text already ends with the parser-safety tail, so ours is only added when it is empty."""
    try:
        characters = list(characters)
        register = get_social_register_text(config, characters, with_tail=not flavor_text)
    except Exception:
        return flavor_text  # never break prompt generation over a style hint
    if not register:
        return flavor_text
    return f"{flavor_text} {register}" if flavor_text else register
