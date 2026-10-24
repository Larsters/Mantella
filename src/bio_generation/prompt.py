"""Prompt for one short, in-lore bio, and parsing of the model's answer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.bio_generation.npc_facts import NpcFacts
from src.bio_generation.quality import BioQuality, is_role_name, word_count

SYSTEM_PROMPT = """You write short character bios for minor NPCs in The Elder Scrolls V: Skyrim (4E 201, during the civil war between the Imperial Legion and the Stormcloaks, with dragons returning).
Rules:
- Stay strictly in lore and consistent with the given race, sex, location, hold and role. Infer the role from the in-game name, voice type and equipment.
- Make the person ordinary and specific: a small, believable life. No world-changing claims: they are not secretly royal, a Dragonborn, a Daedric champion or a famous hero, and they do not know quest secrets. Do not invent relatives or events involving named major characters.
- If the in-game name is a role (e.g. "Whiterun Guard", "Bandit Outlaw"), give them a personal given name fitting their race and use it in the bio. If the in-game name is already a personal name, keep it.
- If an existing bio is given, it is true: do not contradict it or repeat it; add personality, daily life and speech.
- Write in the third person, present tense. Prose only, no lists, no quotation of whole example lines.
- "personality_and_speech" must describe how they talk with 2-3 concrete speech quirks (pet phrases, oaths, dialect, verbal tics) suitable for voice acting.
Answer with ONLY a JSON object: {"given_name": "...", "bio": "...", "personality_and_speech": "..."}"""


@dataclass
class GeneratedBio:
    given_name: str
    bio: str
    personality_and_speech: str

    def as_text(self) -> str:
        text = self.bio.strip()
        if self.personality_and_speech.strip():
            text += "\nPersonality & speech: " + self.personality_and_speech.strip()
        return text


def build_messages(facts: NpcFacts, quality: BioQuality, max_words: int = 110) -> list[dict[str, str]]:
    lines = [f"In-game name: {facts.name}",
             f"Sex: {facts.gender}",
             f"Race: {facts.race}"]
    if facts.voice_type:
        lines.append(f"Voice type: {facts.voice_type}" + (f" (suggests: {facts.voice_hint})" if facts.voice_hint else ""))
    if facts.location:
        lines.append(f"Met at: {facts.location}" + (f" ({facts.hold})" if facts.hold else ""))
    if facts.equipment:
        lines.append(f"Equipment: {facts.equipment}")
    if facts.is_enemy:
        lines.append("Currently hostile to the player.")
    seed = facts.seed
    if seed:
        name_hint = f"; given name starting with '{seed['name_initial']}'" if is_role_name(facts.name) else ""
        lines.append(f"Variety hints (use unless they clash with the facts): temperament {seed['temperament']}; "
                     f"age {seed['age']}; include {seed['life_detail']}{name_hint}.")

    if quality == BioQuality.SHARED:
        task = ("Every person of this kind shares the general description below. Write the INDIVIDUAL: "
                "who this particular one is, so they do not sound like every other " + facts.name + ".")
    elif quality == BioQuality.WEAK:
        task = "The existing bio below is short. Flesh out this person without contradicting it."
    else:
        task = "There is no bio for this person. Write one."
    lines.append(task)
    if facts.existing_bio and quality != BioQuality.MISSING:
        lines.append("Existing bio (true, for context; " + ("general description of the role" if quality == BioQuality.SHARED else "keep consistent") + "):\n"
                     + _truncate_words(facts.existing_bio, 250))
    lines.append(f"Length: bio {max_words // 2}-{max_words} words, personality_and_speech 25-50 words.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


def _truncate_words(text: str, limit: int) -> str:
    words = str(text).split()
    return text if len(words) <= limit else " ".join(words[:limit]) + " ..."


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str | None, min_words: int = 15, max_words: int = 320) -> GeneratedBio | None:
    """Parse the model output; None if it is unusable (so the caller falls back)."""
    if not text or not text.strip():
        return None
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    data = None
    match = _JSON_RE.search(raw)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict):
        bio = str(data.get("bio", "") or "").strip()
        speech = str(data.get("personality_and_speech", "") or "").strip()
        name = str(data.get("given_name", "") or "").strip()
    else:  # plain-text answer: accept it as the bio if it looks like prose
        if "{" in raw:
            return None
        bio, speech, name = raw, "", ""
    total = word_count(bio) + word_count(speech)
    if word_count(bio) < min_words or total > max_words:
        return None
    return GeneratedBio(given_name=name, bio=bio, personality_and_speech=speech)
