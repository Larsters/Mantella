"""Builds the {knowledge_scope} prompt placeholder.

Rule-based, no LLM call. From data Mantella already has about an NPC (name, race and
voice type sent by the game, equipped items, the CSV/override bio) and the current
location, it writes 1-3 sentences on what this person plausibly knows and how they talk.
A farmer should not know Thalmor secrets; a court wizard should not sound like a beggar.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable

from src.knowledge_scope.tables import (
    EQUIPMENT_ROLES, HOLDS, NOT_FROM_NAME, PLACE_HOLDS, RACE_ALIASES, RACES, ROLE_KEYWORDS, ROLES,
    VOICE_ROLES, WHOLE_BIO_ROLES, NOBLE_FAMILY_NAMES, NAME_PREFIX_ROLES,
)

if TYPE_CHECKING:
    from src.character_manager import Character

_BIO_OPENING_CHARS = 300
_FIELD_REGEX = re.compile(r"(?im)^\s*(?:occupation|role|job|profession)\s*:\s*(.+)$")
_HOME_REGEX = re.compile(r"(?im)(?:home location\s*:\s*(.+)$|\blives in ([^.\n]+)|\bresides in ([^.\n]+))")
_OLD_VOICES = ("oldgrumpy", "oldkindly")


@lru_cache(maxsize=None)
def _keyword_regex(keyword: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(keyword) + r"(?:s|es)?\b", re.IGNORECASE)


def normalize_race(raw_race: str | None) -> str | None:
    """Turns the race the game sends ("[Race <DarkElfRace (00013742)>]") or a CSV race
    ("Dunmer", "Dark Elf") into a key of RACES, or None for creatures / unknown races."""
    if not raw_race:
        return None
    race = raw_race
    if "<" in race:
        race = race.split("<", 1)[1].split(" ", 1)[0].split(">", 1)[0]
    race = race.lower().replace(" ", "").replace("_", "")
    race = re.sub(r"vampire$", "", race)
    if race.endswith("race"):
        race = race[:-4]
    return RACE_ALIASES.get(race)


def _match_role(text: str, allowed: frozenset[str] | None = None, excluded: frozenset[str] = frozenset()) -> str | None:
    """Returns the role whose keyword appears EARLIEST in the text (ties: table order)."""
    if not text:
        return None
    best: tuple[int, int, str] | None = None
    for order, (role, keywords) in enumerate(ROLE_KEYWORDS):
        if role in excluded or (allowed is not None and role not in allowed):
            continue
        for keyword in keywords:
            match = _keyword_regex(keyword).search(text)
            if match and (best is None or (match.start(), order) < best[:2]):
                best = (match.start(), order, role)
    return best[2] if best else None


def detect_role(name: str, bio: str, voice_type: str, equipment_names: Iterable[str]) -> str:
    """Returns a key of ROLES. Order of evidence: child voice, name, Occupation/Role lines
    in the bio, the opening of the bio, a few unmistakable roles anywhere in the bio, the
    in-game voice type, equipped items. Falls back to commoner."""
    bio = bio or ""
    voice_role = VOICE_ROLES.get((voice_type or "").lower())
    if voice_role == "child":
        return "child"
    for prefix, prefix_role in NAME_PREFIX_ROLES:
        if (name or "").lower().startswith(prefix):
            return prefix_role
    fields = " ".join(m.group(1) for m in _FIELD_REGEX.finditer(bio))
    for text, kwargs in ((name, {"excluded": NOT_FROM_NAME}),
                         (fields, {}),
                         (bio[:_BIO_OPENING_CHARS], {}),
                         (bio, {"allowed": WHOLE_BIO_ROLES})):
        role = _match_role(text, **kwargs)
        if role:
            return role
    if any(family in (name or "").lower() for family in NOBLE_FAMILY_NAMES):
        return "noble"
    if voice_role:
        return voice_role
    for item in equipment_names:
        item_lower = item.lower()
        for fragment, equipment_role in EQUIPMENT_ROLES:
            if fragment in item_lower:
                return equipment_role
    return "commoner"


def detect_hold(bio: str, location: str | None) -> tuple[str | None, bool]:
    """Returns (hold key, is_home). The bio's home location wins over the current location."""
    for match in _HOME_REGEX.finditer(bio or ""):
        home_text = next(g for g in match.groups() if g)
        hold = _place_to_hold(home_text)
        if hold:
            return hold, True
    hold = _place_to_hold(location or "")
    if hold:
        return hold, False
    return None, False


def _place_to_hold(text: str) -> str | None:
    text = text.lower()
    for fragment, hold in PLACE_HOLDS:
        if fragment in text:
            return hold
    return None


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _equipment_names(character: "Character") -> list[str]:
    names: list[str] = []
    equipment = getattr(character, "equipment", None)
    if not equipment:
        return names
    for slot in ("body", "head", "hands", "feet", "righthand", "lefthand"):
        try:
            item = equipment.get_item(slot)
        except Exception:
            item = None
        if item and isinstance(item.name, str):
            names.append(item.name)
    return names


def build_for_character(character: "Character", location: str | None) -> str:
    """1-3 sentences describing what this NPC knows and how they speak. Empty for the
    player and for creatures whose race is unknown and who have no recognisable role."""
    if character.is_player_character:
        return ""
    name = character.name
    bio = character.bio or ""
    voice = character.in_game_voice_model or ""
    race_key = normalize_race(character.race)
    role_key = detect_role(name, bio, voice, _equipment_names(character))
    if race_key is None and role_key == "commoner":
        return ""  # creature, daedra, dragon or unknown: no sensible scope
    role = ROLES[role_key]
    race = RACES.get(race_key) if race_key else None
    hold_key, is_home = detect_hold(bio, location)
    hold = HOLDS.get(hold_key) if hold_key else None

    gender = getattr(character, "gender", 0)
    subject = "she" if gender == 1 else "he"
    old = (voice.lower().endswith(_OLD_VOICES) or (character.race or "").lower().find("elderrace") >= 0)

    label = role.label
    if race:
        article, noun = label.split(" ", 1)
        descriptor = f"{'old ' if old else ''}{race.noun}".strip() or "common"
        label = f"{_article(descriptor)} {descriptor} {noun}"
    elif old:
        label = f"an old {label.split(' ', 1)[1]}"
    first = f"{name} is {label}"
    if hold:
        first += f" {'of' if is_home else 'in'} {hold.name}, {hold.note}"
    first += "."

    second = f"{subject.capitalize()} knows {role.knows}; of {role.unaware}, only rumours at best."
    third = f"{subject.capitalize()} speaks {role.speech}"
    if race and role_key != "child":
        third += f", {race.speech}"
    third += "."
    return f"{first} {second} {third}"


def build_knowledge_scope(characters: Iterable["Character"], location: str | None, enabled: bool = True) -> str:
    """Text for the {knowledge_scope} placeholder for every NPC in the conversation."""
    if not enabled:
        return ""
    parts = []
    for character in characters:
        try:
            text = build_for_character(character, location)
        except Exception:  # never break prompt generation over a heuristic
            text = ""
        if text:
            parts.append(text)
    return "\n".join(parts)
