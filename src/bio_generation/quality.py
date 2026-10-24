"""Decides whether an NPC's resolved bio is good enough or needs a generated one.

Categories (see docs/features/feat-generated-bios.md for the measurements on skyrim_characters.csv):
- MISSING: the NPC is not in the CSV (Mantella's "You are a male Nord Bandit." one-liner) or the bio is blank.
- SHARED:  the bio is a class template that every individual of that kind shares ("Bandit", "Whiterun Guard",
           46 x "Imperial Soldier"). The lore is good but every guard is the same person.
- WEAK:    a named individual whose bio is shorter than `min_words` words.
- OK:      leave alone.
"""
from __future__ import annotations

import re
from enum import Enum

import pandas as pd


class BioQuality(str, Enum):
    MISSING = "missing"
    SHARED = "shared"
    WEAK = "weak"
    OK = "ok"


class BioScope(str, Enum):
    """Which categories get a generated bio (config value `bio_generation_scope`)."""
    MISSING_ONLY = "Missing only"
    MISSING_AND_WEAK = "Missing and weak"
    ALL = "Missing, weak and shared templates"

    def covers(self, quality: BioQuality) -> bool:
        if quality == BioQuality.OK:
            return False
        if quality == BioQuality.MISSING:
            return True
        if quality == BioQuality.WEAK:
            return self in (BioScope.MISSING_AND_WEAK, BioScope.ALL)
        return self == BioScope.ALL

    @staticmethod
    def from_config(value: str) -> "BioScope":
        for scope in BioScope:
            if scope.value.lower() == str(value).strip().lower():
                return scope
        return BioScope.ALL


# Words that mark an in-game name as a role rather than a person ("Bandit Outlaw", "Whiterun Guard", "Regular Patron").
ROLE_WORDS = (
    r"guards?|bandits?|soldiers?|legionnaires?|stormcloaks?|mercenar(?:y|ies)|thugs?|outlaws?|highwaym[ae]n|"
    r"marauders?|plunderers?|reavers?|lieutenants?|ringleaders?|necromancers?|warlocks?|conjurers?|mages?|"
    r"sorcer(?:er|ess)|witch(?:es)?|cultists?|vampires?|forsworn|thalmor|justiciars?|hunters?|farmers?|miners?|"
    r"laborers?|beggars?|vagrants?|patrons?|revelers?|bards?|travell?ers?|merchants?|couriers?|prisoners?|sailors?|"
    r"dockworkers?|servants?|wench|citizens?|townsfolk|urchins?|priests?|acolytes?|initiates?|apprentices?|archers?|"
    r"warriors?|sentr(?:y|ies)|patrol|agents?|officers?|scouts?|sentinels?|vigilants?|dawnguard|bodyguards?|"
    r"innkeepers?|barkeeps?|cooks?|stablehands?|refugees?|pilgrims?|adventurers?|assassins?|thie(?:f|ves)|"
    r"smugglers?|poachers?|addicts?|guardsm[ae]n|militia|raiders?|brigands?|cutthroats?|skooma"
)
_ROLE_RE = re.compile(rf"\b(?:{ROLE_WORDS})\b", re.IGNORECASE)
# Mantella's fallback for NPCs that are not in the CSV (skyrim.py load_unnamed_npc).
_FALLBACK_RE = re.compile(r"^\s*You are an? (?:male|female)\b.{0,80}\.\s*$", re.IGNORECASE | re.DOTALL)


def is_role_name(name: str) -> bool:
    return bool(_ROLE_RE.search(str(name or "")))


def word_count(text: str | None) -> int:
    return len(str(text or "").split())


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


class TemplateIndex:
    """Index over the character table: which bio texts are class templates shared between different NPC names."""

    def __init__(self, character_df: pd.DataFrame | None):
        self.__template_bios: set[str] = set()
        if character_df is None or character_df.empty or "bio" not in character_df.columns:
            return
        bios = character_df["bio"].fillna("").astype(str).map(_normalise)
        names = character_df["name"].fillna("").astype(str).str.strip().str.lower()
        refs = character_df["ref_id"].fillna("").astype(str).str.strip() if "ref_id" in character_df.columns \
            else pd.Series("", index=character_df.index)
        frame = pd.DataFrame({"bio": bios, "name": names, "ref": refs})
        frame = frame[frame["bio"] != ""]
        # 1) the same bio text used by >= 2 differently named rows ("Bandit", "Bandit Outlaw", ... share one text)
        distinct_names = frame.groupby("bio")["name"].nunique()
        self.__template_bios.update(distinct_names[distinct_names >= 2].index)
        # 2) a role-like name without a ref-id: a template for every placed copy ("Whiterun Guard", "Hunter")
        role_rows = frame[(frame["ref"] == "") & frame["name"].map(is_role_name)]
        self.__template_bios.update(role_rows["bio"])

    def is_template_bio(self, bio: str) -> bool:
        return _normalise(bio) in self.__template_bios

    @property
    def size(self) -> int:
        return len(self.__template_bios)


def classify(bio: str | None, is_generic_npc: bool, index: TemplateIndex, min_words: int) -> BioQuality:
    text = str(bio or "").strip()
    if is_generic_npc or not text or _FALLBACK_RE.match(text):
        return BioQuality.MISSING
    if index.is_template_bio(text):
        return BioQuality.SHARED
    if word_count(text) < min_words:
        return BioQuality.WEAK
    return BioQuality.OK
