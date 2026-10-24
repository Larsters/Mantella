"""Data for src/social_register: instruction texts and detection tables.

Texts are templates: {names} = NPC name(s), {s} = verb -s for one NPC.
Parser safety (see src/speech_flavor): no colons, brackets, asterisks or quotes in any text.
Sources for names and terms: docs/features/feat-social-register.md (UESP pages cited there).
"""

STREET_TEXT = (
    "{names} talk{s} street, low-class slang in every reply like coin, mark, fence, a job, the Flagon, "
    "sugar and skooma, light fingers, tin-hats for guards, skeever-bait, milk-drinker, Shor's bones. "
    "Short and blunt, never refined."
)

KHAJIIT_TEXT = (
    "{names} speak{s} Khajiit street talk, calling self this one, Khajiit or by name, never I. "
    "Sly haggling, coin, a mark, the fence, moon sugar and skooma, by Jone and Jode, "
    "shadows hide you, walk on warm sands."
)

TAIL = "Stay understandable and in character, use actions normally. No bracketed explanations, asterisks or colons mid-sentence."

# Known low-class / underworld NPCs (lower case). Verified on UESP: Skyrim:Beggars, Skyrim:Thieves_Guild_(faction).
STREET_NAMES: frozenset[str] = frozenset({
    # beggars
    "brenuin", "edda", "snilf", "angrenor once-honored", "noster eagle-eye", "degaine", "silda the unseen",
    "dervenin", "narfi", "lucia",
    # Thieves Guild members
    "brynjolf", "delvin mallory", "vex", "vekel the man", "tonilia", "dirge", "sapphire", "niruin", "rune",
    "cynric endell", "thrynn", "etienne rarnis", "garthar", "ravyn imyan", "vipir the fleet", "mercer frey",
    "arnskar ember-master", "herluin lothaire", "syndus", "vanryth gatharian", "glover mallory", "maul",
    # fences
    "gulum-ei", "mallus maccius", "niranye",
})

# In-game voice types (GetVoiceType, lower case). MaleDrunk is used by Brenuin, Degaine and Narfi (skyrim_characters.csv).
# MaleCoward/FemaleCoward are NOT used: too many ordinary townsfolk share them.
STREET_VOICES: frozenset[str] = frozenset({"maledrunk"})
KHAJIIT_VOICES: frozenset[str] = frozenset({"malekhajiit", "femalekhajiit"})

# Keywords searched in the first 300 characters of the bio (whole words, plural -s/-es allowed).
BIO_KEYWORDS: tuple[str, ...] = (
    "beggar", "vagrant", "homeless", "thief", "thieves", "thieves guild", "pickpocket", "cutpurse", "burglar",
    "fence", "smuggler", "skooma", "moon sugar", "drunkard", "drunk", "ratway", "ragged flagon", "the warrens",
    "dock worker", "dockworker", "dockhand", "laborer", "labourer", "miner", "smelter worker", "farmhand",
    "farm hand", "stable hand", "stablehand", "lumberjack", "servant", "orphan", "urchin", "bandit",
)

# Never street register through detection (an explicit override still wins):
# guards and soldiers have their own military jargon override; Lydia interprets jargon via her own prompt.
EXCLUDED_NAMES: frozenset[str] = frozenset({"lydia"})
EXCLUDED_NAME_FRAGMENTS: tuple[str, ...] = ("guard", "soldier", "legionnaire")
EXCLUDED_VOICES: frozenset[str] = frozenset({"maleguard", "femaleguard", "malesoldier", "femalesoldier"})
