"""Rule tables for the {knowledge_scope} prompt placeholder.

Everything here is plain data: who a person is (role), where they live (hold) and
what their people are like (race) mapped to short phrases about what they would
plausibly know and how they talk. Lore follows the vanilla start of Skyrim (4E 201,
civil war undecided). Keep phrases short: they end up in every system prompt.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    label: str      # "a city guard"
    knows: str      # what this person knows well
    unaware: str    # what they only know by rumour, or not at all
    speech: str     # register / manner of speaking


# Shared fallbacks
_COMMON_UNAWARE = "the war's high politics, the Thalmor's plans, magic and far-off provinces"

ROLES: dict[str, Role] = {
    "jarl": Role("a jarl",
                 "the politics of the hold, the other jarls, the war and the Thalmor's demands on Skyrim",
                 "the daily lives of common folk beyond what reports say",
                 "with the authority of a ruler, formal and used to being obeyed"),
    "court_wizard": Role("a court wizard",
                         "magic, enchanting, ancient lore and the court's affairs",
                         "the trades and gossip of common folk",
                         "in an educated, precise and slightly condescending way"),
    "steward": Role("a jarl's steward or advisor",
                    "the hold's coffers, laws, trade and court intrigue",
                    "the details of soldiering and the wilds",
                    "formally and carefully, like an official"),
    "housecarl": Role("a sworn housecarl",
                      "weapons, soldiering, the hold's court and the duty of a sworn shield",
                      "magic, scholarship and court secrets",
                      "plainly and dutifully, in short clipped sentences"),
    "thalmor": Role("a Thalmor agent",
                    "Thalmor policy, the ban on Talos worship and who in Skyrim breaks it",
                    "the everyday life of Nords, which they hold in contempt",
                    "with cold, superior courtesy that hides a threat"),
    "legion": Role("an Imperial soldier",
                   "Legion orders, drills, the war from the Empire's side and the roads they patrol",
                   "court secrets, magic and the Thalmor's real plans",
                   "with soldierly discipline, loyal to the Empire and General Tullius"),
    "stormcloak": Role("a Stormcloak soldier",
                       "the rebellion, Ulfric's cause, Talos worship and the war from the Stormcloak side",
                       "court secrets, magic and Imperial plans",
                       "with fierce Nord pride, contemptuous of the Empire and the elves"),
    "guard": Role("a guard",
                  "local law, bounties, the roads, the jarl's orders and whoever passes the gates",
                  "court secrets, magic and the Thalmor's plans",
                  "bluntly, in clipped soldier's talk, often bored or suspicious"),
    "companion": Role("a member of the Companions",
                      "fighting, monster hunting, honour and the Companions' traditions",
                      "magic, politics and court affairs",
                      "boldly and proudly, like a seasoned warrior"),
    "mage": Role("a mage",
                 "spells, the schools of magic, enchanting and arcane lore",
                 "farming, trades and street gossip",
                 "in an educated, sometimes lecturing way"),
    "priest": Role("a priest",
                   "the Divines, temple rites, healing and the dead",
                   "warfare, crime and court intrigue",
                   "gently and reverently, often invoking the gods"),
    "merchant": Role("a merchant",
                     "goods, prices, trade routes, coin and who owes whom",
                     _COMMON_UNAWARE,
                     "persuasively, always with an eye on profit"),
    "blacksmith": Role("a blacksmith",
                       "metals, weapons, armour, forging and who buys what",
                       _COMMON_UNAWARE,
                       "plainly and practically, proud of good work"),
    "innkeeper": Role("an innkeeper",
                      "rumours, travellers, local gossip, food and drink",
                      "the war's high politics, magic and court secrets",
                      "in a friendly, worldly way, quick with gossip"),
    "bard": Role("a bard",
                 "songs, old tales, history and the gossip of taverns",
                 "the details of war, trade and magic",
                 "theatrically, fond of a turn of phrase"),
    "alchemist": Role("an alchemist",
                      "herbs, potions, poisons and ingredients",
                      "the war's high politics and court secrets",
                      "precisely, with a healer's or herbalist's knowledge"),
    "farmer": Role("a farmer",
                   "crops, livestock, weather, harvests and the families nearby",
                   _COMMON_UNAWARE,
                   "simply and plainly, worried about the harvest and the next tax"),
    "miner": Role("a miner",
                  "ore, mines, hard labour and the mine owners",
                  _COMMON_UNAWARE,
                  "roughly and tiredly, a working man's or woman's talk"),
    "hunter": Role("a hunter",
                   "the wilds, animals, tracks, trails and dangerous places",
                   "politics, magic and city affairs",
                   "tersely, more at home in the woods than in a crowd"),
    "labourer": Role("a labourer",
                     "their trade, the local folk and the price of bread",
                     _COMMON_UNAWARE,
                     "plainly and simply"),
    "sailor": Role("a sailor or dockworker",
                   "ships, the sea, ports, cargo and smuggling rumours",
                   "magic, court affairs and inland politics",
                   "roughly, with a dockside directness"),
    "beggar": Role("a beggar",
                   "the streets, who gives coin, which guards are cruel and the city's underside",
                   "politics beyond street rumour, magic and the doings of the rich",
                   "humbly or bitterly, always angling for a coin"),
    "noble": Role("a member of a wealthy family",
                  "family business, property, local politics, feuds and who matters in town",
                  "hard labour, the wilds and magic",
                  "with the confidence of money, often looking down on the poor"),
    "thief": Role("a thief",
                  "fences, guard rotations, locks, the underworld and who has coin",
                  "magic and court affairs beyond what is worth stealing",
                  "slyly and guardedly, never saying more than needed"),
    "bandit": Role("a bandit",
                   "the roads, ambush spots, hideouts, loot and which guards to avoid",
                   "court affairs, magic and anything beyond survival",
                   "with crude threats and contempt for honest folk"),
    "forsworn": Role("a Forsworn warrior",
                     "the Reach, the old ways, Hagravens and every Nord wrong done to their people",
                     "Imperial and Nord politics, which they despise",
                     "with fierce hatred for Nords and outsiders"),
    "mercenary": Role("a sellsword",
                      "fighting, contracts, roads and who pays well",
                      "court secrets and magic",
                      "bluntly and transactionally"),
    "vampire": Role("a vampire",
                    "blood, the night, old secrets and the weaknesses of mortals",
                    "the day-to-day concerns of the living",
                    "with predatory, ancient calm"),
    "dawnguard": Role("a vampire hunter of the Dawnguard",
                      "vampires, their weaknesses and the old Dawnguard order",
                      "court affairs and magic beyond what kills vampires",
                      "grimly and with a hunter's focus"),
    "child": Role("a child",
                  "games, family, friends and what the grown-ups say at home",
                  "almost everything outside home and street",
                  "like a child, in short, simple and curious sentences"),
    "commoner": Role("a common townsperson",
                     "their own trade, family, neighbours and local gossip",
                     _COMMON_UNAWARE,
                     "plainly, like ordinary folk"),
}

# Ordered (role, keywords) list. First match wins, so specific roles come first.
# Keywords are matched as whole words (an optional plural "s"/"es" is allowed) against
# the role text: the name first, then Occupation/Role lines, then the opening of the bio.
# Keep them specific to avoid false positives.
ROLE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("child", ("is a child", "is a young boy", "is a young girl", "orphan")),
    ("jarl", ("is the jarl", "is jarl", "jarl of", "high king", "high queen")),
    ("court_wizard", ("court wizard", "court mage")),
    ("thalmor", ("thalmor", "justiciar")),
    ("steward", ("steward", "advisor to the jarl", "the jarl's advisor")),
    ("housecarl", ("housecarl",)),
    ("forsworn", ("forsworn", "briarheart")),
    ("dawnguard", ("dawnguard",)),
    ("vampire", ("vampire", "volkihar")),
    ("companion", ("the companions", "jorrvaskr")),
    ("legion", ("imperial soldier", "legionnaire", "legate", "imperial legion soldier", "penitus oculatus")),
    ("stormcloak", ("stormcloak soldier", "member of the stormcloaks", "soldier in the stormcloak army", "fights for the stormcloaks")),
    ("mercenary", ("mercenary", "sellsword", "hireling", "bodyguard")),
    ("guard", ("guard",)),
    ("bandit", ("bandit", "marauder", "highwayman", "outlaw", "brigand", "thug")),
    ("thief", ("thieves guild", "thief", "pickpocket", "fence")),
    ("priest", ("priest", "priestess", "temple of", "vigilant of stendarr", "monk")),
    ("mage", ("mage", "wizard", "sorcerer", "sorceress", "college of winterhold", "necromancer", "conjurer", "apprentice")),
    ("bard", ("bard",)),
    ("alchemist", ("alchemist", "apothecary", "herbalist")),
    ("blacksmith", ("blacksmith", "smith", "armorer", "armourer", "warmaiden")),
    ("innkeeper", ("innkeeper", "bartender", "barkeep", "tavern owner", "owner of the inn", "owns the inn", "serving wench")),
    ("merchant", ("merchant", "trader", "shopkeeper", "general goods", "businessman", "businesswoman", "tradeswoman", "tradesman", "caravan", "vendor", "jeweler", "fletcher")),
    ("farmer", ("farmer", "farmhand", "farm", "miller", "rancher")),
    ("miner", ("miner", "mine")),
    ("hunter", ("hunter", "huntress", "trapper")),
    ("sailor", ("sailor", "fisherman", "fisherwoman", "dockworker", "dock worker", "ship captain", "fishery")),
    ("labourer", ("lumberjack", "woodcutter", "labourer", "laborer", "stablehand", "stable master", "carriage driver", "courier", "cook", "servant", "maid", "tanner")),
    ("beggar", ("beggar", "homeless", "vagrant")),
    ("noble", ("battle-born clan", "gray-mane clan", "grey-mane clan", "black-briar", "silver-blood family", "noble", "nobleman", "noblewoman", "wealthy", "aristocrat", "brother of jarl", "sister of jarl", "son of jarl", "daughter of jarl")),
]

# Roles that may be picked up anywhere in a bio (last resort before voice and equipment).
# Only roles whose keyword is unlikely to describe someone else in the bio.
WHOLE_BIO_ROLES: frozenset[str] = frozenset({"housecarl", "court_wizard"})

# Clan names that mark a wealthy family when nothing more specific (a trade) was found.
NOBLE_FAMILY_NAMES: tuple[str, ...] = ("battle-born", "gray-mane", "grey-mane", "black-briar", "silver-blood")

# Name prefixes that make the role unambiguous.
NAME_PREFIX_ROLES: tuple[tuple[str, str], ...] = (("jarl ", "jarl"), ("high king ", "jarl"), ("high queen ", "jarl"))

# Roles that are never inferred from the NPC's name alone (family names are not jobs).
NOT_FROM_NAME: frozenset[str] = frozenset({"noble"})

# In-game voice types (from GetVoiceType) that hint at a role when nothing else matched.
VOICE_ROLES: dict[str, str] = {
    "maleguard": "guard",
    "femaleguard": "guard",
    "malesoldier": "guard",
    "femalesoldier": "guard",
    "malebandit": "bandit",
    "malechild": "child",
    "femalechild": "child",
    "malewarlock": "mage",
    "dlc1malevampire": "vampire",
    "dlc1femalevampire": "vampire",
    "maledrunk": "beggar",
    "malecondescending": "noble",
    "femalecondescending": "noble",
}

# Equipment item name fragments (lower case) that hint at a role. Checked in order.
EQUIPMENT_ROLES: list[tuple[str, str]] = [
    ("thalmor", "thalmor"),
    ("stormcloak", "stormcloak"),
    ("imperial", "legion"),
    ("guard's armor", "guard"),
    ("forsworn", "forsworn"),
    ("dawnguard", "dawnguard"),
    ("blacksmith's apron", "blacksmith"),
    ("miner's clothes", "miner"),
    ("farm clothes", "farmer"),
    ("tavern clothes", "innkeeper"),
    ("chef's", "labourer"),
    ("priest", "priest"),
    ("monk", "priest"),
    ("novice robes", "mage"),
    ("apprentice robes", "mage"),
    ("adept robes", "mage"),
    ("expert robes", "mage"),
    ("master robes", "mage"),
    ("mage", "mage"),
    ("necromancer", "mage"),
    ("fine clothes", "noble"),
    ("rags", "beggar"),
    ("roughspun", "beggar"),
    ("fur armor", "hunter"),
    ("hide armor", "hunter"),
]


@dataclass(frozen=True)
class Race:
    noun: str       # "Nord"
    speech: str     # cultural colour of their speech


RACES: dict[str, Race] = {
    "nord": Race("Nord", "with Nord bluntness and pride, swearing by Talos, Shor or Ysmir"),
    "imperial": Race("Imperial", "in a measured, lawful Imperial manner, invoking the Nine"),
    "breton": Race("Breton", "in a quick, articulate Breton manner"),
    "redguard": Race("Redguard", "with Redguard pride and directness"),
    "altmer": Race("Altmer", "with High Elf precision and a sense of superiority"),
    "bosmer": Race("Bosmer", "with quick, lively Wood Elf wit"),
    "dunmer": Race("Dunmer", "with dry Dark Elf sarcasm and words like s'wit and n'wah, wary of Nords"),
    "orc": Race("Orc", "gruffly, honour-bound in the way of the strongholds"),
    "khajiit": Race("Khajiit", "as Khajiit do, calling themselves this one or Khajiit, sly and courteous"),
    "argonian": Race("Argonian", "with formal, slightly odd Argonian phrasing"),
    "reachman": Race("Reachman", "with the hard, guarded manner of the Reach"),
    "elder": Race("", "slowly, in the set ways of the old"),
}

# Race id aliases. Keys are lower-case with spaces removed and a trailing "race" stripped.
RACE_ALIASES: dict[str, str] = {
    "nord": "nord", "imperial": "imperial", "breton": "breton", "redguard": "redguard",
    "altmer": "altmer", "highelf": "altmer",
    "bosmer": "bosmer", "woodelf": "bosmer",
    "dunmer": "dunmer", "darkelf": "dunmer",
    "orc": "orc", "orsimer": "orc",
    "khajiit": "khajiit", "argonian": "argonian",
    "reachman": "reachman",
    "elder": "elder",  # ElderRace: old Nords/Imperials; no nationality, only age
}


@dataclass(frozen=True)
class Hold:
    name: str       # "Whiterun Hold"
    note: str       # one clause of local situation at the game's start


HOLDS: dict[str, Hold] = {
    "whiterun": Hold("Whiterun Hold", "Jarl Balgruuf's hold, keeping out of the war for now"),
    "haafingar": Hold("Haafingar", "ruled from Solitude by Jarl Elisif, the Empire's seat in Skyrim"),
    "eastmarch": Hold("Eastmarch", "ruled from Windhelm by Ulfric Stormcloak, heart of the rebellion"),
    "rift": Hold("the Rift", "ruled from Riften by Jarl Laila Law-Giver, where the Black-Briars pull the strings"),
    "reach": Hold("the Reach", "ruled from Markarth by Jarl Igmund, troubled by the Forsworn"),
    "falkreath": Hold("Falkreath Hold", "the forested hold of Jarl Siddgeir"),
    "hjaalmarch": Hold("Hjaalmarch", "the marsh hold of Jarl Idgrod Ravencrone in Morthal"),
    "pale": Hold("the Pale", "ruled from Dawnstar by Jarl Skald the Elder"),
    "winterhold": Hold("Winterhold", "Jarl Korir's half-ruined hold, wary of the College mages"),
    "solstheim": Hold("Solstheim", "a Dunmer island under Raven Rock's council, far from Skyrim's war"),
}

# Place-name fragments (lower case) mapped to holds. Checked against the location
# the game sends and the bio's home-location text.
PLACE_HOLDS: list[tuple[str, str]] = [
    ("whiterun", "whiterun"), ("dragonsreach", "whiterun"), ("jorrvaskr", "whiterun"),
    ("bannered mare", "whiterun"), ("riverwood", "whiterun"), ("rorikstead", "whiterun"),
    ("breezehome", "whiterun"), ("warmaiden", "whiterun"), ("sleeping giant", "whiterun"),
    ("solitude", "haafingar"), ("blue palace", "haafingar"), ("dragon bridge", "haafingar"),
    ("winking skeever", "haafingar"), ("castle dour", "haafingar"), ("haafingar", "haafingar"),
    ("windhelm", "eastmarch"), ("candlehearth", "eastmarch"), ("kynesgrove", "eastmarch"),
    ("palace of the kings", "eastmarch"), ("gray quarter", "eastmarch"), ("eastmarch", "eastmarch"),
    ("riften", "rift"), ("ivarstead", "rift"), ("ragged flagon", "rift"), ("bee and barb", "rift"),
    ("mistveil", "rift"), ("shor's stone", "rift"), ("the rift", "rift"),
    ("markarth", "reach"), ("understone", "reach"), ("karthwasten", "reach"), ("cidhna", "reach"),
    ("silver-blood inn", "reach"), ("the reach", "reach"),
    ("falkreath", "falkreath"), ("helgen", "falkreath"),
    ("morthal", "hjaalmarch"), ("highmoon hall", "hjaalmarch"), ("hjaalmarch", "hjaalmarch"),
    ("dawnstar", "pale"), ("windpeak", "pale"), ("the pale", "pale"),
    ("winterhold", "winterhold"), ("college of winterhold", "winterhold"), ("frozen hearth", "winterhold"),
    ("raven rock", "solstheim"), ("skaal", "solstheim"), ("solstheim", "solstheim"),
]
