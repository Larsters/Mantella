"""Speech flavour presets as data.

Add a preset by adding an entry to PRESETS. Each preset has
- label: shown in the Mantella UI selection
- intensities: one style instruction per intensity (light / medium / heavy), each <= ~80 tokens
- reduced: one short line for NPCs who get the preset at "light" strength inside a multi-NPC prompt
- none_line: line for NPCs in a multi-NPC prompt who keep their own speech (no flavour)
- race_levels: per-race strength relative to the configured intensity (used when scope = "By race")
  0 = configured intensity, -1 = one step lighter (but at least light), None = no flavour
- tts_respellings: (written form, spoken form) pairs applied ONLY to the text sent to an English TTS voice
  (Piper/espeak-ng 'en'), so subtitles keep proper German. Matched case-insensitively at word boundaries,
  longer phrases first.
"""
from dataclasses import dataclass, field

INTENSITIES: list[str] = ["light", "medium", "heavy"]

# Shared tail: German must never cost playability or break Mantella's output parsers
# (':' switches speaker / stops generation, '*' '(' '[' mark narration, quotes are stripped).
_GERMAN_TAIL = ("Stay clear in English and in character, use actions normally. "
                "No bracketed translations, asterisks or colons mid-sentence.")


@dataclass(frozen=True)
class SpeechFlavorPreset:
    id: str
    label: str
    intensities: dict[str, str]
    reduced: str
    none_line: str
    race_levels: dict[str, int | None] = field(default_factory=dict)
    tts_respellings: list[tuple[str, str]] = field(default_factory=list)


GERMAN_MIX = SpeechFlavorPreset(
    id="german_mix",
    label="German mix",
    intensities={
        "light": ("Speech style: English with a German accent, about one German word per reply "
                  "(Ja, Genau, Natürlich, Ach so, Mein Freund, Wunderbar). " + _GERMAN_TAIL),
        "medium": ("Speech style: English like a native German, one or two German words per reply "
                   "(Ja, Natürlich, Genau, Mein Freund, Ach so, Moment mal, Wunderbar, rarely Scheiße), sometimes "
                   "German word order or false friends (I become a sweetroll). " + _GERMAN_TAIL),
        "heavy": ("Speech style: Denglisch, half of each sentence in simple German (Ja, Natürlich, Genau, Mein Freund, "
                  "Was ist los, Jawohl, Scheiße), German word order, false friends, silly compound nouns "
                  "(Drachenproblem). Key facts in English. " + _GERMAN_TAIL),
    },
    reduced="{names} use only about one German word per reply.",
    none_line="{names} speak normally, without German.",
    race_levels={
        "nord": 0,
        "imperial": -1, "breton": -1, "redguard": -1, "orc": -1,
        "altmer": -1, "bosmer": -1, "dunmer": -1,
        "khajiit": None, "argonian": None,
    },
    tts_respellings=[
        ("Danke schön", "Dahnkeh shern"),
        ("Sehr gut", "Zair goot"),
        ("Mein Gott", "Mine Gott"),
        ("Mein Freund", "Mine froynd"),
        ("Moment mal", "Moh ment mahl"),
        ("Ach so", "Akh zo"),
        ("Scheiße", "Shy-sa"),
        ("Scheisse", "Shy-sa"),
        ("Natürlich", "Nah-toor-lish"),
        ("Gemütlich", "Geh-moot-lish"),
        ("Mädchen", "Maid-shen"),
        ("Achtung", "Ahkh-toong"),
        ("Jawohl", "Yah-vohl"),
        ("Genau", "Geh-now"),
        ("Freunde", "Froyndeh"),
        ("Freund", "Froynd"),
        ("Danke", "Dahnkeh"),
        ("schön", "shern"),
        ("Prost", "Prohst"),
        ("Nein", "Nine"),
        ("Mein", "Mine"),
        ("Ja", "Yah"),
        ("Ach", "Akh"),
    ],
)

PRESETS: dict[str, SpeechFlavorPreset] = {p.id: p for p in [GERMAN_MIX]}

NONE_LABEL = "None"


def preset_labels() -> list[str]:
    """Options for the UI selection: 'None' first, then every preset label."""
    return [NONE_LABEL] + [p.label for p in PRESETS.values()]


def find_preset(value: str | None) -> SpeechFlavorPreset | None:
    """Accepts a preset id ('german_mix') or label ('German mix'), case-insensitive. None/'None'/'' -> None."""
    if not value:
        return None
    key = str(value).strip().lower()
    for preset in PRESETS.values():
        if key in (preset.id.lower(), preset.label.lower(), preset.label.lower().replace(' ', '_')):
            return preset
    return None
