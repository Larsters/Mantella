"""Opt-in self-aware companion (default companion: Lydia). Config group "Companion", default off.

When ``companion_enabled`` is on and the player talks to a companion ONE-ON-ONE:
1. The one-on-one prompt is replaced by the companion's own template
   ``data/<game>/companion/<name>_prompt.txt`` (a copy in the Mantella user folder under the same
   relative path takes precedence). Missing template = the normal prompt is kept.
2. ``{about_player}`` is filled with the user's ``about_me.txt`` (facts about the REAL player).
   In every other situation (other NPCs, multi-NPC, radiant, feature off) it is an empty string.
   In shared multi-NPC / radiant prompts the companion's own memory summaries are also left out
   (unless ``companion_group_memories``), because they can contain what was said in private.
3. The ``RememberAboutPlayer`` server tool (registered in ``src.actions.server_tools``) is offered
   and appends new facts to about_me.txt. It is never forwarded to the game.
Every file access is fail-safe: a missing or unreadable file never raises.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src import utils
from src.actions import server_tools

logger = utils.get_logger()

ABOUT_ME_FILENAME = "about_me.txt"
ABOUT_PLAYER_PLACEHOLDER = "{about_player}"
REMEMBER_ACTION_ID = "mantella_remember_about_player"
LEARNED_SECTION = "[Learned in game]"
MAX_FACT_CHARS = 200
MAX_ABOUT_ME_CHARS = 4000  # keeps the prompt bounded even if the file grows large
EMPTY_ABOUT_PLAYER = ("Nothing is known about them yet beyond what they tell you. "
                      "Get to know them naturally, one question at a time.")


@dataclass
class CompanionSettings:
    enabled: bool = False
    names: str = "Lydia"
    about_me_path: str = ""
    group_memories: bool = False
    save_folder: str = ""


settings = CompanionSettings()
_write_lock = threading.Lock()
_FIELD_ONLY = re.compile(r"^[^:\[\]]{1,40}:\s*$")  # an unfilled "Key:" placeholder line
_SECTION = re.compile(r"^\[.+\]$")
_DATE_PREFIX = re.compile(r"^-\s*\d{4}-\d{2}-\d{2}:\s*")


def configure(config) -> None:
    """Read the Companion settings from a ConfigLoader (called whenever a Context is created)."""
    global settings
    new = CompanionSettings(save_folder=getattr(config, "save_folder", "") or "")
    definitions = getattr(config, "definitions", None)
    if definitions is not None:
        for attr, key, getter in (("enabled", "companion_enabled", "get_bool_value"),
                                  ("names", "companion_names", "get_string_value"),
                                  ("about_me_path", "companion_about_me_path", "get_string_value"),
                                  ("group_memories", "companion_group_memories", "get_bool_value")):
            try:
                setattr(new, attr, getattr(definitions, getter)(key))
            except Exception:
                pass  # setting not defined (eg older config) -> keep default
    settings = new


# --- who is the companion ---------------------------------------------------------------------

def companion_names() -> set[str]:
    return {n.strip().lower() for n in (settings.names or "").split(",") if n.strip()}


def is_companion(character) -> bool:
    name = getattr(character, "name", None)
    return settings.enabled and bool(name) and name.strip().lower() in companion_names()


def is_companion_one_on_one(characters) -> bool:
    """True if the player talks to exactly one NPC and that NPC is a companion."""
    if not settings.enabled or characters is None or not characters.contains_player_character():
        return False
    npcs = characters.get_non_player_characters()
    return len(npcs) == 1 and is_companion(npcs[0])


# --- about_me.txt -----------------------------------------------------------------------------

def about_me_path() -> Path:
    if settings.about_me_path:
        return Path(settings.about_me_path).expanduser()
    base = settings.save_folder or os.path.join(os.path.expanduser("~"), "Documents", "My Games", "Mantella")
    return Path(base) / ABOUT_ME_FILENAME


def read_about_me(path: Path | None = None) -> str:
    """Return the filled-in content of about_me.txt: no '#' notes, no empty 'Key:' lines,
    no empty sections. Returns '' if the file is missing or unreadable."""
    path = path or about_me_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return ""

    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or _FIELD_ONLY.match(line):
            continue
        if _SECTION.match(line):
            sections.append((line, []))
        else:
            sections[-1][1].append(line)

    out: list[str] = []
    for header, lines in sections:
        if lines:
            if header:
                out.append(header)
            out.extend(lines)
    text = "\n".join(out)
    if len(text) > MAX_ABOUT_ME_CHARS:
        text = text[-MAX_ABOUT_ME_CHARS:]  # keep the newest (bottom) part
    return text


def about_player_text(characters) -> str:
    """Value of {about_player}: about_me.txt in a one-on-one with the companion, else ''."""
    if not is_companion_one_on_one(characters):
        return ""
    return read_about_me() or EMPTY_ABOUT_PLAYER


# --- prompt -----------------------------------------------------------------------------------

def _game_folder(game) -> str:
    base = getattr(game, "base_game", game)
    return "Fallout4" if "fallout" in str(getattr(base, "display_name", base)).lower() else "Skyrim"


def get_prompt_template(name: str, game_folder: str = "Skyrim") -> str | None:
    relative = Path("data") / game_folder / "companion" / f"{name.strip().lower().replace(' ', '_')}_prompt.txt"
    candidates = ([Path(settings.save_folder) / relative] if settings.save_folder else []) + [Path(utils.resolve_path()) / relative]
    for p in candidates:
        try:
            if p.is_file():
                text = p.read_text(encoding="utf-8")
                if text.strip():
                    return text
        except Exception as e:
            logger.warning(f"Could not read companion prompt {p}: {e}")
    return None


def select_prompt(default_prompt: str, characters, game=None) -> str:
    """The prompt template to use: the companion's own template in a one-on-one with her, else
    the default. Guarantees the companion's prompt contains {about_player}."""
    if not is_companion_one_on_one(characters):
        return default_prompt
    companion = characters.get_non_player_characters()[0]
    template = get_prompt_template(companion.name, _game_folder(game)) or default_prompt
    if ABOUT_PLAYER_PLACEHOLDER not in template:
        template += "\n\n# The real person playing\n" + ABOUT_PLAYER_PLACEHOLDER
    return template


def characters_for_shared_memories(characters: list) -> list:
    """Characters whose summaries may go into the prompt. In shared (multi-NPC/radiant) prompts the
    companion's private summaries are dropped unless companion_group_memories is on."""
    if len(characters) <= 1 or settings.group_memories:
        return characters
    return [c for c in characters if not is_companion(c)]


# --- RememberAboutPlayer ----------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def clean_fact(fact) -> str:
    if not isinstance(fact, str):
        return ""
    fact = " ".join(fact.split()).strip().strip("\"'").strip()
    fact = fact.lstrip("#[ -").strip()
    if len(fact) > MAX_FACT_CHARS:
        fact = fact[:MAX_FACT_CHARS].rsplit(" ", 1)[0].rstrip(",;") + "..."
    return fact


def remember_fact(fact, path: Path | None = None, now: datetime | None = None) -> tuple[bool, str]:
    """Append one fact to about_me.txt under [Learned in game] (keep that section last).
    Skips empty and duplicate facts. Returns (saved, message). Never raises."""
    path = path or about_me_path()
    fact = clean_fact(fact)
    if not fact:
        return False, "empty fact"
    try:
        with _write_lock:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            new_norm = _normalize(fact)
            for line in existing.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                old_norm = _normalize(_DATE_PREFIX.sub("", line))
                if old_norm == new_norm or (len(new_norm) >= 12 and new_norm in old_norm):
                    return False, "duplicate"
            addition = "\n" if existing and not existing.endswith("\n") else ""
            if LEARNED_SECTION not in existing:
                addition += f"\n{LEARNED_SECTION}\n"
            addition += f"- {(now or datetime.now()).strftime('%Y-%m-%d')}: {fact}\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(addition)
        logger.log(23, f"RememberAboutPlayer: saved a new fact to {path}")
        return True, "saved"
    except Exception as e:
        logger.error(f"RememberAboutPlayer: could not write {path}: {e}")
        return False, str(e)


def _remember_handler(arguments: dict, characters) -> str:
    saved, message = remember_fact(arguments.get("fact"))
    return "Saved to your notes about the player." if saved else f"Not saved ({message})."


def _remember_available(context) -> bool:
    return is_companion_one_on_one(getattr(context, "npcs_in_conversation", None))


server_tools.register(REMEMBER_ACTION_ID, _remember_handler, _remember_available)
