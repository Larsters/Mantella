"""Checks for the shipped character override presets (Lydia, Adrianne Avenicci).

The presets live in data/Skyrim/character_override_presets/ and are installed by copying them into
the user's ~/Documents/My Games/Mantella/data/Skyrim/character_overrides/ folder.
"""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.config.config_loader import ConfigLoader
from src.games.skyrim import Skyrim

PRESETS_DIR = Path("data/Skyrim/character_override_presets")
LORE_PRESETS = {
    "lydia.json": ("Lydia", "0A2C8E", "Nord"),
    "adrianne_avenicci.json": ("Adrianne Avenicci", "013BB9", "Imperial"),
}
SELF_AWARE_PRESET = PRESETS_DIR / "optional" / "lydia_self_aware.json"
ALL_PRESETS = [PRESETS_DIR / name for name in LORE_PRESETS] + [SELF_AWARE_PRESET]
BIO_TOKEN_BUDGET = 600  # the bio is sent with every turn


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def skyrim_with_overrides(tmp_path, default_config: ConfigLoader):
    def _make(*preset_paths: Path) -> Skyrim:
        default_config.mod_path_base = str(tmp_path)
        default_config.save_folder = str(tmp_path)
        override_dir = tmp_path / "data" / "Skyrim" / "character_overrides"
        override_dir.mkdir(parents=True, exist_ok=True)
        for p in preset_paths:
            shutil.copy(p, override_dir / p.name)
        return Skyrim(default_config)
    return _make


@pytest.mark.parametrize("path", ALL_PRESETS, ids=lambda p: p.name)
def test_preset_is_ascii_json_without_format_braces(path: Path):
    raw = path.read_bytes()
    # gameable.py opens override files without an explicit encoding (cp1252 on Windows), so stay ASCII
    raw.decode("ascii")
    data = json.loads(raw)
    assert data["name"] and data["base_id"] and data["race"] and data["bio"]
    # bios are inserted into prompts; keep them free of format placeholders
    assert "{" not in data["bio"] and "}" not in data["bio"]


@pytest.mark.parametrize("path", ALL_PRESETS, ids=lambda p: p.name)
def test_preset_bio_token_budget(path: Path):
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # encoding file not cached / no network
        pytest.skip(f"cl100k_base unavailable: {e}")
    assert len(enc.encode(_load(path)["bio"])) <= BIO_TOKEN_BUDGET


@pytest.mark.parametrize("file_name", LORE_PRESETS.keys())
def test_lore_presets_replace_bio_and_keep_other_fields(skyrim_with_overrides, file_name: str):
    name, base_id, race = LORE_PRESETS[file_name]
    preset = _load(PRESETS_DIR / file_name)
    skyrim = skyrim_with_overrides(PRESETS_DIR / file_name)

    df = skyrim.character_df
    rows = df[df["name"] == name]
    assert len(rows) == 1, "override must update the existing row, not add a new one"
    row = rows.iloc[0]
    assert row["bio"] == preset["bio"]
    assert row["base_id"] == base_id and row["race"] == race
    # voice and per-character model fields come from the CSV untouched
    assert row["voice_model"] in ("FemaleEvenToned", "FemaleCommander")
    assert pd.isna(row["model"]) or row["model"] == ""

    info = skyrim.load_external_character_info(base_id, name, race, 1, row["voice_model"])
    assert info.bio == preset["bio"]
    assert info.llm_model == ""


def test_adrianne_preset_facts():
    bio = _load(PRESETS_DIR / "adrianne_avenicci.json")["bio"]
    assert "Proventus Avenicci" in bio and "Aventus" not in bio
    for name in ("Ulfberth", "Eorlund Gray-Mane", "Idolaf Battle-Born", "Sigurd"):
        assert name in bio
    assert "Barter" in bio and "Never Follows" in bio


def test_lydia_preset_rules():
    bio = _load(PRESETS_DIR / "lydia.json")["bio"]
    for word in ("Follow", "Wait", "Unfollow", "Inventory", "never her Thane"):
        assert word in bio
    assert "Style reference only" in bio


def test_self_aware_variant_sets_per_character_model(skyrim_with_overrides):
    preset = _load(SELF_AWARE_PRESET)
    skyrim = skyrim_with_overrides(SELF_AWARE_PRESET)
    info = skyrim.load_external_character_info("0A2C8E", "Lydia", "Nord", 1, "FemaleEvenToned")
    assert info.bio == preset["bio"]
    assert info.llm_service == "OpenAI"
    assert info.llm_model == preset["model"]


def test_self_aware_variant_is_not_loaded_from_subfolder(skyrim_with_overrides, tmp_path):
    """The opt-in variant sits in optional/, a subfolder the override loader ignores."""
    lore = _load(PRESETS_DIR / "lydia.json")
    optional_dir = tmp_path / "data" / "Skyrim" / "character_overrides" / "optional"
    optional_dir.mkdir(parents=True)
    shutil.copy(SELF_AWARE_PRESET, optional_dir / SELF_AWARE_PRESET.name)
    skyrim = skyrim_with_overrides(PRESETS_DIR / "lydia.json")
    df = skyrim.character_df
    assert df[df["name"] == "Lydia"].iloc[0]["bio"] == lore["bio"]


def test_csv_bio_fixes():
    df = pd.read_csv("data/Skyrim/skyrim_characters.csv", encoding="utf-8-sig", dtype=str)
    adrianne = df[df["name"] == "Adrianne Avenicci"].iloc[0]["bio"]
    assert "Proventus Avenicci: Her father" in adrianne
    assert "Aventus Avenicci" not in adrianne
    lydia = df[df["name"] == "Lydia"].iloc[0]["bio"]
    assert "Personality:" in lydia
