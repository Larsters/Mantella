import pytest

from src.character_manager import Character
from src.games.equipment import Equipment, EquipmentItem
from src.knowledge_scope import (
    build_for_character, build_knowledge_scope, detect_hold, detect_role, normalize_race,
)


def make_npc(name="Guard", race="[Race <NordRace (00013746)>]", bio="", voice="MaleEvenToned",
             gender=0, items: dict[str, str] | None = None, is_player=False) -> Character:
    equipment = Equipment({slot: EquipmentItem(item) for slot, item in (items or {}).items()})
    return Character("0", "0", name, gender, race, is_player, bio, False, False, 0, False,
                     voice, voice, voice, voice, "en", equipment, {})


@pytest.mark.parametrize("raw,expected", [
    ("[Race <NordRace (00013746)>]", "nord"),
    ("[Race <DarkElfRace (00013742)>]", "dunmer"),
    ("[Race <HighElfRace (00013743)>]", "altmer"),
    ("[Race <WoodElfRace (00013749)>]", "bosmer"),
    ("[Race <OrcRace (00013747)>]", "orc"),
    ("[Race <KhajiitRaceVampire (00088845)>]", "khajiit"),
    ("[Race <ElderRace (00067CD8)>]", "elder"),
    ("Dunmer", "dunmer"),
    ("Dark Elf", "dunmer"),
    ("Orsimer", "orc"),
    ("[Race <DragonRace (00012E82)>]", None),
    ("", None),
    (None, None),
])
def test_normalize_race(raw, expected):
    assert normalize_race(raw) == expected


@pytest.mark.parametrize("name,bio,voice,items,expected", [
    ("Whiterun Guard", "", "MaleGuard", None, "guard"),
    ("Jarl Ulfric Stormcloak", "", "MaleUniqueUlfric", None, "jarl"),
    ("Farengar Secret-Fire", "Role: Court wizard of Whiterun. Advises the jarl on magical matters.", "MaleEvenTonedAccented", None, "court_wizard"),
    ("Brenuin", "Brenuin is a Redguard man who lives in Whiterun. He is a homeless beggar and drunkard, asking merchants for money.", "MaleDrunk", None, "beggar"),
    ("Lars Battle-Born", "Lars Battle-Born is a child who lives with his family in Whiterun.", "MaleChild", None, "child"),
    ("Eorlund Gray-Mane", "Race: Nord\nRole: Eorlund is widely considered to be the best blacksmith in all of Skyrim.", "MaleNordCommander", None, "blacksmith"),
    ("Idolaf Battle-Born", "Physical Appearance: Adult man.", "MaleBrute", None, "noble"),
    ("Lydia", "Physical Appearance: Adult woman.\n" + "x" * 400 + " Jarl Balgruuf appointed Lydia to serve as their personal housecarl.", "FemaleEvenToned", None, "housecarl"),
    ("Ancano", "Ancano is a Thalmor operative.", "MaleUniqueAncano", None, "thalmor"),
    ("Hulda", "Occupation: Hulda is the owner and innkeeper of the Bannered Mare.", "FemaleNord", None, "innkeeper"),
    # A mage's "image" or "damage" must not match "mage"
    ("Some Person", "She took damage to her image.", "FemaleCommoner", None, "commoner"),
    # Falls back to voice, then equipment
    ("Mysterious Stranger", "", "MaleBandit", None, "bandit"),
    ("Old Woman", "", "FemaleOldKindly", {"body": "Farm Clothes"}, "farmer"),
    ("Robed Figure", "", "MaleEvenToned", {"body": "Thalmor Robes"}, "thalmor"),
    ("Nobody", "", "MaleEvenToned", None, "commoner"),
])
def test_detect_role(name, bio, voice, items, expected):
    assert detect_role(name, bio, voice, list((items or {}).values())) == expected


def test_detect_hold_prefers_bio_home_over_current_location():
    assert detect_hold("Home Location: Windhelm. Often at Candlehearth Hall.", "Whiterun") == ("eastmarch", True)
    assert detect_hold("He lives in Riften with his mother.", None) == ("rift", True)
    assert detect_hold("", "Dragonsreach") == ("whiterun", False)
    assert detect_hold("", "Skyrim") == (None, False)
    assert detect_hold("", None) == (None, False)


def test_build_guard_scope_mentions_role_race_hold_and_limits():
    text = build_for_character(make_npc("Whiterun Guard", voice="MaleGuard"), "Whiterun")
    assert text.startswith("Whiterun Guard is a Nord guard in Whiterun Hold")
    assert "bounties" in text
    assert "Thalmor" in text  # listed among the things a guard knows only by rumour
    assert "He speaks" in text
    assert 1 <= text.count(".") <= 4
    assert ":" not in text


def test_build_uses_female_pronoun_and_old_age():
    text = build_for_character(make_npc("Old Woman", voice="FemaleOldKindly", gender=1, items={"body": "Farm Clothes"}), "Riverwood")
    assert text.startswith("Old Woman is an old Nord farmer in Whiterun Hold")
    assert "She knows crops" in text


def test_farmer_and_court_wizard_differ():
    farmer = build_for_character(make_npc("Farmer", bio="", voice="MaleCommoner"), "Whiterun")
    wizard = build_for_character(make_npc("Farengar Secret-Fire", bio="Role: Court wizard of Whiterun."), "Whiterun")
    assert "Thalmor's plans" in farmer and "only rumours" in farmer
    assert "magic" in wizard.split(";")[0]  # magic is something the wizard KNOWS
    assert "educated" in wizard and "plainly" in farmer


def test_child_has_no_race_speech():
    text = build_for_character(make_npc("Lars Battle-Born", voice="MaleChild"), "Whiterun")
    assert "like a child" in text
    assert "Talos" not in text


def test_empty_for_player_and_creatures():
    assert build_for_character(make_npc(is_player=True), "Whiterun") == ""
    assert build_for_character(make_npc("Dragon", race="[Race <DragonRace (00012E82)>]", voice="CrDragonVoice"), "Whiterun") == ""


def test_build_knowledge_scope_multi_and_disabled():
    npcs = [make_npc("Whiterun Guard", voice="MaleGuard"),
            make_npc("Ysolda", bio="Ysolda is a keen tradeswoman.", gender=1),
            make_npc(is_player=True)]
    text = build_knowledge_scope(npcs, "Whiterun")
    lines = text.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("Whiterun Guard is")
    assert lines[1].startswith("Ysolda is a Nord merchant")
    assert build_knowledge_scope(npcs, "Whiterun", enabled=False) == ""


def test_build_knowledge_scope_never_raises():
    class Broken:
        is_player_character = False
        @property
        def name(self):
            raise RuntimeError("boom")
    assert build_knowledge_scope([Broken()], "Whiterun") == ""
