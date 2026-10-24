"""Lore corpus: loading and validating the JSON lore notes.

A corpus folder contains any number of ``*.json`` files shaped like::

    {"attribution": "...", "notes": [ {note}, ... ]}

Each note:
    id        unique id (later folders override earlier ones with the same id)
    title     short display title
    scope     place | faction | person | race | event | topic
    aliases   names that clearly refer to the note (matched as whole phrases)
    keywords  weaker related terms (scored with BM25)
    parent    optional id of the containing place (Dragonsreach -> Whiterun -> Whiterun Hold)
    places    optional ids of places where the note is especially relevant
    priority  optional float (default 1.0) used when picking scene notes for a place
    text      2-4 sentences of common in-world knowledge
    spoiler   true if it reveals a quest twist (excluded unless spoilers are allowed)
    gate      optional quest name after which the note becomes common knowledge
    source    attribution URL (UESP, CC BY-SA 2.5)
    disabled  optional; true removes a shipped note (useful in a personal override file)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src import utils

logger = utils.get_logger()

VALID_SCOPES = {"place", "faction", "person", "race", "event", "topic"}


@dataclass(frozen=True)
class LoreNote:
    id: str
    title: str
    scope: str
    text: str
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    parent: str | None = None
    places: tuple[str, ...] = ()
    priority: float = 1.0
    spoiler: bool = False
    gate: str | None = None
    source: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @staticmethod
    def from_dict(d: dict) -> "LoreNote":
        missing = [k for k in ("id", "title", "scope", "text") if not d.get(k)]
        if missing:
            raise ValueError(f"lore note {d.get('id', '?')!r} is missing {missing}")
        scope = str(d["scope"]).lower()
        if scope not in VALID_SCOPES:
            raise ValueError(f"lore note {d['id']!r} has unknown scope {scope!r}")
        known = {"id", "title", "scope", "text", "aliases", "keywords", "parent", "places",
                 "priority", "spoiler", "gate", "source", "disabled"}
        return LoreNote(
            id=str(d["id"]),
            title=str(d["title"]),
            scope=scope,
            text=" ".join(str(d["text"]).split()),
            aliases=tuple(str(a) for a in d.get("aliases", []) if str(a).strip()),
            keywords=tuple(str(k) for k in d.get("keywords", []) if str(k).strip()),
            parent=d.get("parent") or None,
            places=tuple(d.get("places", []) or ()),
            priority=float(d.get("priority", 1.0)),
            spoiler=bool(d.get("spoiler", False)),
            gate=d.get("gate") or None,
            source=str(d.get("source", "")),
            extra={k: v for k, v in d.items() if k not in known},
        )


def load_corpus(folders: list[str]) -> dict[str, LoreNote]:
    """Loads all notes from the given folders, in order. Later folders override earlier ones by id.
    Invalid files or notes are logged and skipped; a missing folder is ignored."""
    notes: dict[str, LoreNote] = {}
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        for file_name in sorted(os.listdir(folder)):
            if not file_name.lower().endswith(".json"):
                continue
            path = os.path.join(folder, file_name)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Lore: could not read {path}: {e}")
                continue
            raw_notes = data.get("notes", []) if isinstance(data, dict) else data
            for raw in raw_notes:
                try:
                    if raw.get("disabled"):
                        notes.pop(str(raw.get("id")), None)
                        continue
                    note = LoreNote.from_dict(raw)
                except Exception as e:
                    logger.warning(f"Lore: skipping invalid note in {path}: {e}")
                    continue
                notes[note.id] = note
    return notes
