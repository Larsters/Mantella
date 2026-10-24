"""Turns what the game sends about an NPC into plain facts for the bio prompt.

The game sends (per actor): name, base-id, ref-id, gender, race ("[Race <NordRace (00013746)>]"),
voice type ("[VoiceType <MaleGuard (0002F7C3)>]"), equipment, relationship rank, combat/enemy flags
and custom values; the conversation context carries the location. Class, faction and level are NOT sent,
so the role is inferred from the name, voice type and equipment.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

_RACE_NAMES = {
    "nord": "Nord", "imperial": "Imperial", "breton": "Breton", "redguard": "Redguard",
    "darkelf": "Dunmer (Dark Elf)", "highelf": "Altmer (High Elf)", "woodelf": "Bosmer (Wood Elf)",
    "orc": "Orc (Orsimer)", "khajiit": "Khajiit", "argonian": "Argonian",
    "elder": "Imperial (elderly)", "snowelf": "Falmer (Snow Elf)",
}

# Settlements / landmarks -> hold. Only the obvious ones; the model knows the rest.
_HOLDS = {
    "whiterun": "Whiterun Hold", "dragonsreach": "Whiterun Hold", "riverwood": "Whiterun Hold",
    "rorikstead": "Whiterun Hold", "bannered mare": "Whiterun Hold", "jorrvaskr": "Whiterun Hold",
    "solitude": "Haafingar", "blue palace": "Haafingar", "dragon bridge": "Haafingar", "winking skeever": "Haafingar",
    "windhelm": "Eastmarch", "palace of the kings": "Eastmarch", "kynesgrove": "Eastmarch", "candlehearth": "Eastmarch",
    "riften": "the Rift", "mistveil": "the Rift", "ivarstead": "the Rift", "shor's stone": "the Rift", "bee and barb": "the Rift",
    "markarth": "the Reach", "understone keep": "the Reach", "karthwasten": "the Reach", "old hroldan": "the Reach",
    "falkreath": "Falkreath Hold", "helgen": "Falkreath Hold",
    "dawnstar": "the Pale", "windpeak": "the Pale",
    "morthal": "Hjaalmarch", "highmoon": "Hjaalmarch",
    "winterhold": "Winterhold", "frozen hearth": "Winterhold",
    "raven rock": "Solstheim", "solstheim": "Solstheim", "skaal": "Solstheim",
}

# Fragments of vanilla voice-type names -> what they suggest about the person.
_VOICE_HINTS = [
    ("child", "a child (keep everything age-appropriate)"),
    ("oldgrumpy", "old and grumpy"), ("oldkindly", "old and kindly"), ("old", "elderly"),
    ("youngeager", "young and eager"), ("coward", "timid, easily frightened"), ("brute", "big and brutish"),
    ("commander", "used to giving orders"), ("condescending", "condescending"), ("sultry", "smooth, sultry voice"),
    ("drunk", "a heavy drinker"), ("haughty", "haughty"), ("shrill", "shrill voice"), ("guard", "a guard"),
    ("soldier", "a soldier"), ("bandit", "a bandit"), ("forsworn", "Forsworn"), ("commoner", "a commoner"),
    ("eventoned", "calm, even-toned"), ("warlock", "a spellcaster"),
]

_TEMPERAMENTS = [
    "dry and sardonic", "earnest and dutiful", "gloomy and fatalistic", "cheerful despite everything",
    "suspicious of strangers", "boastful", "quietly kind", "short-tempered", "nosy and gossipy",
    "weary and practical", "pious", "greedy but honest about it", "superstitious", "proud and touchy",
    "lazy and work-shy", "nervous", "blunt and loud", "melancholic and poetic", "calculating", "good-humoured",
]
_LIFE_DETAILS = [
    "a family member they worry about", "a debt they cannot pay", "a hobby (dice, carving, fishing or the like)",
    "an old injury", "a rival in the same line of work", "a favourite tavern or drink", "a past they do not talk about",
    "a small ambition", "a grudge against a specific local", "a pet or animal they care for", "homesickness",
    "a fear (spiders, magic, deep water or the like)", "a sweetheart", "a lucky charm", "religious devotion to one of the Divines",
]
_AGES = ["young adult", "in their late twenties", "in their thirties", "in their forties", "in their fifties", "grizzled veteran"]
_INITIALS = "ABDEFGHIJKLMNORSTUVY"


@dataclass
class NpcFacts:
    name: str
    ref_id: str
    base_id: str
    gender: str
    race: str
    voice_type: str = ""
    voice_hint: str = ""
    location: str = ""
    hold: str = ""
    equipment: str = ""
    existing_bio: str = ""
    is_enemy: bool = False
    seed: dict[str, str] = field(default_factory=dict)


def parse_race(race: str) -> str:
    """'[Race <DarkElfRace (00013742)>]' -> 'Dunmer (Dark Elf)'; plain names pass through."""
    raw = str(race or "")
    match = re.search(r"<\s*([A-Za-z_]+?)(?:Race)?(?:Vampire)?\s*(?:\(|>)", raw)
    key = match.group(1) if match else raw.replace("Race", "").strip()
    return _RACE_NAMES.get(key.lower().replace(" ", ""), key or "unknown race")


_CREATURE_RE = re.compile(r"dwarven|dwemer|sphere|spider|centurion|dragon|atronach|skeleton|draugr|wolf|bear|horse|"
                          r"dog|spriggan|troll|giant|mammoth|chaurus|skeever|sabre|frostbite|mudcrab|wisp|netch|"
                          r"riekling|seeker|lurker|gargoyle|deer|elk|goat|cow|chicken|fox|rabbit|slaughterfish", re.IGNORECASE)


def is_creature_race(race: str) -> bool:
    """Automatons and animals have no use for a personal bio (the CSV has e.g. 'Dwemer Sphere')."""
    raw = str(race or "")
    match = re.search(r"<\s*([A-Za-z_]+)", raw)
    return bool(_CREATURE_RE.search(match.group(1) if match else raw))


def parse_voice_type(voice: str) -> str:
    """'[VoiceType <MaleGuard (0002F7C3)>]' -> 'MaleGuard'."""
    raw = str(voice or "")
    match = re.search(r"<\s*([A-Za-z0-9_]+)", raw)
    return match.group(1) if match else raw.strip()


def voice_hint(voice_type: str) -> str:
    key = voice_type.lower()
    hints: list[str] = []
    for fragment, hint in _VOICE_HINTS:
        if fragment in key and not any(fragment in h for h in hints):
            hints.append(hint)
            if fragment.startswith("old"):
                break
    return ", ".join(hints[:2])


def hold_for(location: str) -> str:
    key = str(location or "").lower()
    for fragment, hold in _HOLDS.items():
        if fragment in key:
            return hold
    return ""


def variety_seed(key: str) -> dict[str, str]:
    """Deterministic per NPC so the prompt is stable, varied across NPCs so guards don't clone each other."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return {
        "temperament": _TEMPERAMENTS[digest[0] % len(_TEMPERAMENTS)],
        "life_detail": _LIFE_DETAILS[digest[1] % len(_LIFE_DETAILS)],
        "age": _AGES[digest[2] % len(_AGES)],
        "name_initial": _INITIALS[digest[3] % len(_INITIALS)],
    }


def facts_from_character(character: Any, location: str, existing_bio: str, seed_key: str) -> NpcFacts:
    """Build facts from a src.character_manager.Character (duck-typed so tests can pass simple objects)."""
    voice_type = parse_voice_type(getattr(character, "in_game_voice_model", "") or "")
    equipment = ""
    try:
        equipment_obj = getattr(character, "equipment", None)
        if equipment_obj is not None:
            equipment = equipment_obj.get_equipment_description(character.name)
    except Exception:
        equipment = ""
    return NpcFacts(
        name=str(character.name),
        ref_id=str(getattr(character, "ref_id", "") or ""),
        base_id=str(getattr(character, "base_id", "") or ""),
        gender="female" if getattr(character, "gender", 0) == 1 else "male",
        race=parse_race(getattr(character, "race", "")),
        voice_type=voice_type,
        voice_hint=voice_hint(voice_type),
        location=str(location or ""),
        hold=hold_for(location),
        equipment=equipment,
        existing_bio=existing_bio,
        is_enemy=bool(getattr(character, "is_enemy", False)),
        seed=variety_seed(seed_key),
    )
