"""Speech flavour: NPCs answer in English sprinkled with another language (start: German mix).

Public API (both are no-ops while `speech_flavor = None`, the upstream default):
- get_speech_flavor_text(config, characters) -> str   fills the `{speech_flavor}` prompt placeholder
- prepare_for_tts(text, config) -> str                 respells German words for English TTS voices (Piper)

Per-NPC override: add `"speech_flavor": "<value>"` to the NPC's JSON in
`<Mantella save folder>/data/Skyrim/character_overrides/` (matched by "name", plus "base_id" if given).
Values: "none" / "off", an intensity ("light" | "medium" | "heavy"), a preset id or label ("german_mix"),
or "<preset>:<intensity>" ("german_mix:heavy"). An override ignores the race scaling.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from src.speech_flavor.presets import INTENSITIES, PRESETS, SpeechFlavorPreset, find_preset

SCOPE_ALL = "All NPCs"
SCOPE_BY_RACE = "By race"
SCOPES = [SCOPE_BY_RACE, SCOPE_ALL]

_RACE_KEYS: list[tuple[str, str]] = [
    # order matters: 'darkelf' before 'elf', 'elder' (Imperial elders) before nothing else
    ("darkelf", "dunmer"), ("dunmer", "dunmer"),
    ("highelf", "altmer"), ("altmer", "altmer"),
    ("woodelf", "bosmer"), ("bosmer", "bosmer"),
    ("argonian", "argonian"), ("khajiit", "khajiit"),
    ("nord", "nord"), ("imperial", "imperial"), ("elder", "imperial"),
    ("breton", "breton"), ("redguard", "redguard"), ("orc", "orc"),
]

_OFF_VALUES = {"none", "off", "no", "false", "disabled", "0"}


# ---------------------------------------------------------------- config helpers
def _cfg(config: Any, key: str, default: Any) -> Any:
    value = getattr(config, key, default)
    return default if value is None else value


def _intensity_index(value: Any) -> int:
    v = str(value or "").strip().lower()
    return INTENSITIES.index(v) if v in INTENSITIES else INTENSITIES.index("medium")


def normalize_race(race: str | None) -> str | None:
    """'[Race <NordRace (00013746)>]', 'NordRaceVampire', 'Nord', 'Dark Elf' -> 'nord' / 'dunmer' / ... / None."""
    if not race:
        return None
    flat = re.sub(r"[^a-z]", "", str(race).lower())
    for needle, key in _RACE_KEYS:
        if needle in flat:
            return key
    return None


def parse_override(value: Any) -> tuple[bool, SpeechFlavorPreset | None, str | None]:
    """-> (disabled, preset or None, intensity or None)."""
    v = str(value or "").strip()
    if not v:
        return False, None, None
    if v.lower() in _OFF_VALUES:
        return True, None, None
    preset, intensity = None, None
    for part in re.split(r"[:,/ ]+", v):
        p = part.strip().lower()
        if not p:
            continue
        if p in INTENSITIES:
            intensity = p
        elif p in _OFF_VALUES:
            return True, None, None
        else:
            preset = find_preset(p) or preset
    if preset is None and intensity is None:
        preset = find_preset(v)
    return False, preset, intensity


# ---------------------------------------------------------------- overrides
_override_cache: dict[str, tuple[float, dict[str, list[tuple[str, str]]]]] = {}


def _overrides_folder(config: Any) -> str | None:
    save_folder = getattr(config, "save_folder", None)
    if not save_folder:
        return None
    game = getattr(config, "game", None)
    game_folder = "Skyrim"
    try:
        if game is not None and "FALLOUT" in game.base_game.name:
            game_folder = "Fallout4"
    except Exception:
        pass
    return os.path.join(save_folder, "data", game_folder, "character_overrides")


def load_overrides(folder: str | None) -> dict[str, list[tuple[str, str]]]:
    """{lower name: [(base_id, speech_flavor value), ...]} from all JSON override files that set `speech_flavor`.
    Cached per folder until a file in it changes."""
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
            if not isinstance(entry, dict) or "speech_flavor" not in entry or not entry.get("name"):
                continue
            result.setdefault(str(entry["name"]).strip().lower(), []).append(
                (str(entry.get("base_id", "") or ""), str(entry["speech_flavor"])))
    _override_cache[folder] = (stamp, result)
    return result


def _norm_id(value: str) -> str:
    return str(value or "").strip().lower().removeprefix("0x").lstrip("0")


def find_override(overrides: dict[str, list[tuple[str, str]]], name: str, base_id: str = "") -> str | None:
    entries = overrides.get(str(name or "").strip().lower(), [])
    fallback = None
    for entry_base_id, value in entries:
        if entry_base_id and _norm_id(entry_base_id) == _norm_id(base_id):
            return value
        if not entry_base_id:
            fallback = value
    return fallback


# ---------------------------------------------------------------- resolution
def resolve(config: Any, name: str, race: str | None, base_id: str = "",
            overrides: dict[str, list[tuple[str, str]]] | None = None) -> tuple[SpeechFlavorPreset | None, int | None]:
    """Effective (preset, intensity index) for one NPC; (None, None) = no flavour."""
    preset = find_preset(_cfg(config, "speech_flavor", "None"))
    level: int | None = _intensity_index(_cfg(config, "speech_flavor_intensity", "medium"))

    override = find_override(overrides or {}, name, base_id)
    if override is not None:
        disabled, o_preset, o_intensity = parse_override(override)
        if disabled:
            return None, None
        preset = o_preset or preset or (PRESETS["german_mix"] if o_intensity else None)
        if preset is None:
            return None, None
        if o_intensity:
            return preset, INTENSITIES.index(o_intensity)
        return preset, level  # override names only a preset: configured intensity, no race scaling

    if preset is None:
        return None, None
    if str(_cfg(config, "speech_flavor_scope", SCOPE_BY_RACE)).strip().lower() == SCOPE_BY_RACE.lower():
        race_key = normalize_race(race)
        if race_key in preset.race_levels:
            delta = preset.race_levels[race_key]
            if delta is None:
                return None, None
            level = max(0, level + delta)
    return preset, level


def _join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def build_text(npcs: list[tuple[str, SpeechFlavorPreset | None, int | None]]) -> str:
    """npcs: [(name, preset, level)] -> placeholder text."""
    flavoured = [(n, p, l) for n, p, l in npcs if p is not None and l is not None]
    if not flavoured:
        return ""
    top_name, top_preset, top_level = max(flavoured, key=lambda x: x[2])
    text = top_preset.intensities[INTENSITIES[top_level]]
    if len(npcs) == 1 or all((p is top_preset and l == top_level) for _, p, l in npcs):
        return text
    strongest = [n for n, p, l in flavoured if p is top_preset and l == top_level]
    reduced = [n for n, p, l in flavoured if not (p is top_preset and l == top_level)]
    plain = [n for n, p, l in npcs if p is None or l is None]
    parts = [f"Only {_join_names(strongest)} speak{'s' if len(strongest) == 1 else ''} like this.", text]
    if reduced:
        parts.append(top_preset.reduced.format(names=_join_names(reduced)))
    if plain:
        parts.append(top_preset.none_line.format(names=_join_names(plain)))
    return " ".join(parts)


def get_speech_flavor_text(config: Any, characters: Iterable[Any]) -> str:
    """Text for `{speech_flavor}`. `characters` are the NPCs in the conversation (player excluded by caller)."""
    if find_preset(_cfg(config, "speech_flavor", "None")) is None and not _has_any_override(config):
        return ""
    overrides = load_overrides(_overrides_folder(config))
    npcs = []
    for c in characters:
        if getattr(c, "is_player_character", False):
            continue
        preset, level = resolve(config, c.name, getattr(c, "race", None), getattr(c, "base_id", ""), overrides)
        npcs.append((c.name, preset, level))
    return build_text(npcs)


def _has_any_override(config: Any) -> bool:
    return bool(load_overrides(_overrides_folder(config)))


# ---------------------------------------------------------------- TTS
_respell_cache: dict[str, list[tuple[re.Pattern, str]]] = {}


def _respell_patterns(preset: SpeechFlavorPreset) -> list[tuple[re.Pattern, str]]:
    if preset.id not in _respell_cache:
        pairs = sorted(preset.tts_respellings, key=lambda p: -len(p[0]))
        _respell_cache[preset.id] = [(re.compile(r"(?<![\w'])" + re.escape(w) + r"(?![\w'])", re.IGNORECASE), s)
                                     for w, s in pairs]
    return _respell_cache[preset.id]


def respell_for_tts(text: str, preset: SpeechFlavorPreset) -> str:
    """German spelling -> English-voice-friendly respelling. Unknown ß becomes ss (espeak-ng 'en' handles ss;
    other umlauts are left as is: espeak approximates them reasonably)."""
    def keep_case(spoken: str, match: re.Match) -> str:
        return spoken[0].upper() + spoken[1:] if match.group(0)[:1].isupper() else spoken[0].lower() + spoken[1:]
    for pattern, spoken in _respell_patterns(preset):
        text = pattern.sub(lambda m, s=spoken: keep_case(s, m), text)
    return text.replace("ß", "ss")


def prepare_for_tts(text: str, config: Any) -> str:
    """Hook for English TTS (Piper). Active only when a preset is selected and `speech_flavor_tts_respell` is on."""
    if not _cfg(config, "speech_flavor_tts_respell", True):
        return text
    preset = find_preset(_cfg(config, "speech_flavor", "None"))
    if preset is None:
        return text
    return respell_for_tts(text, preset)
