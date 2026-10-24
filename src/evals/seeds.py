"""Hand-written seed scenarios (docs/prompt-audit.md section 3) so evals work before real transcripts exist.

Each scenario is turned into the same request shape the server would send: the upstream Skyrim prompt
template filled with the NPC's real bio from data/Skyrim/skyrim_characters.csv, the tools generated from
data/actions/*.json (filtered by conversation type, with entity lists like FunctionManager adds), and
Mantella's default llm_params. `prompt_vars` are kept so a candidate can swap in a new prompt template.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from copy import deepcopy
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PARAMS = {'max_tokens': 250, 'stop': ['#']}
MULTI_NPC_MIN_MAX_TOKENS = 250
PLAYER = 'Eirik'
PLAYER_EQUIPMENT = 'wears Steel Armor, Steel Boots and Steel Gauntlets and uses Steel Sword.'

# expect_tools: at least one of these should be called; forbid_tools: none of these may be called.
SCENARIOS: list[dict[str, Any]] = [
    {
        'id': 'seed-01-guard-civil-war', 'type': 'single', 'npcs': ['Whiterun Guard'],
        'location': 'Whiterun', 'hour': 10, 'day': 12, 'weather': 'The sky is clear and the sun is bright.',
        'equipment': {'Whiterun Guard': 'wears Whiterun Guard\'s Armor and Whiterun Shield and uses Steel Sword.'},
        'nearby': ['Carlotta Valentia'],
        'history': [],
        'user': 'Whose side is Whiterun on in the war?',
        'expected': '1-3 gruff spoken sentences. Whiterun is neutral under Jarl Balgruuf; the guard is loyal to the '
                    'Jarl and wary of both Ulfric and the Legion. No "arrow in the knee" meme, no parroted bio lines. No tool needed.',
        'expect_tools': [], 'forbid_tools': ['Attack', 'Follow', 'Brawl'],
    },
    {
        'id': 'seed-02-belethor-haggle', 'type': 'single', 'npcs': ['Belethor'],
        'location': "Belethor's General Goods", 'hour': 14, 'day': 12, 'weather': '',
        'equipment': {'Belethor': 'wears Fine Clothes and Boots.'},
        'nearby': [],
        'history': [
            ('user', f'*{PLAYER} enters the shop.* Good day, Belethor.'),
            ('assistant', 'Everything\'s for sale, my friend. Everything! If I had a sister, I\'d sell her in a second.'),
        ],
        'user': 'Come on, give me half off everything and I\'ll buy the lot.',
        'expected': 'Sly, sarcastic merchant. Must not promise a discount the game cannot honour (Mantella cannot change prices). '
                    'Opening Barter is fine only if the player wants to see wares; otherwise no tool.',
        'expect_tools': [], 'forbid_tools': ['Attack', 'Brawl', 'Follow'],
    },
    {
        'id': 'seed-03-lydia-wait', 'type': 'single', 'npcs': ['Lydia'],
        'location': 'Bleak Falls Barrow', 'hour': 16, 'day': 14, 'weather': 'Snow falls steadily from a grey sky.',
        'equipment': {'Lydia': 'wears Steel Plate Armor and Steel Plate Boots and uses Steel Sword and Iron Shield.'},
        'nearby': [],
        'history': [
            ('user', 'Lydia, stay close. This place feels wrong.'),
            ('assistant', 'I\'m right behind you, my Thane.'),
        ],
        'user': 'Wait here and guard the entrance. I\'ll go ahead alone.',
        'expected': 'Calls Wait with source [Lydia] and, in the same reply, a short loyal acknowledgement ("As you wish, my Thane."). '
                    'May voice concern briefly. No Follow.',
        'expect_tools': ['Wait'], 'forbid_tools': ['Follow', 'Attack', 'Teleport'],
    },
    {
        'id': 'seed-04-balgruuf-insult', 'type': 'single', 'npcs': ['Balgruuf the Greater'],
        'location': 'Dragonsreach', 'hour': 11, 'day': 12, 'weather': '',
        'equipment': {'Balgruuf the Greater': 'wears Jarl Balgruuf\'s Clothes.'},
        'nearby': ['Irileth', 'Proventus Avenicci'],
        'history': [],
        'user': 'You\'re a coward, Balgruuf. Hiding behind your walls while Ulfric and the Legion decide your fate.',
        'expected': 'Cold, dignified rebuke from a Jarl; may warn the player or tell them to leave. Words alone do not '
                    'justify Attack. No meta talk.',
        'expect_tools': [], 'forbid_tools': ['Attack', 'Brawl', 'Follow'],
    },
    {
        'id': 'seed-05-misheard-stt', 'type': 'single', 'npcs': ['Adrianne Avenicci'],
        'location': 'Warmaiden\'s', 'hour': 9, 'day': 12, 'weather': 'The sky is clear and the sun is bright.',
        'equipment': {'Adrianne Avenicci': 'wears Blacksmith\'s Apron and Boots and uses Blacksmith\'s Hammer.'},
        'nearby': ['Ulfberth War-Bear'],
        'history': [
            ('user', 'Hello there.'),
            ('assistant', 'Welcome to Warmaiden\'s. Weapons, armor, whatever you need, I can make it.'),
        ],
        'user': 'I\'m going to gag in.',
        'expected': 'The input is a speech-to-text error. Good reply: asks the player to repeat ("What was that?"), '
                    'no offence taken, no emote or other action.',
        'expect_tools': [], 'forbid_tools': ['Emote', 'Attack', 'Flee', 'Brawl'],
    },
    {
        'id': 'seed-06-market-multi', 'type': 'multi', 'npcs': ['Nazeem', 'Belethor'],
        'location': 'Whiterun', 'hour': 13, 'day': 12, 'weather': 'The sky is clear and the sun is bright.',
        'equipment': {'Nazeem': 'wears Fine Clothes and Fine Boots.', 'Belethor': 'wears Fine Clothes and Boots.'},
        'nearby': ['Carlotta Valentia'],
        'history': [],
        'user': 'Belethor tells me you still owe him for an amulet, Nazeem.',
        'expected': '1-2 speakers, each 1-2 sentences, each line prefixed with the full speaker name ("Nazeem: ..."). '
                    'Nazeem condescending and evasive, Belethor greedy. No long round-robin. No tool needed.',
        'expect_tools': [], 'forbid_tools': ['Attack', 'Brawl'],
    },
    {
        'id': 'seed-07-radiant-guard-adrianne', 'type': 'radiant', 'npcs': ['Whiterun Guard', 'Adrianne Avenicci'],
        'location': 'Whiterun', 'hour': 15, 'day': 12, 'weather': 'Clouds drift lazily across the sky.',
        'equipment': {'Whiterun Guard': 'wears Whiterun Guard\'s Armor and uses Steel Sword.',
                      'Adrianne Avenicci': 'wears Blacksmith\'s Apron.'},
        'nearby': [],
        'history': [],
        'user': None,  # radiant_start_prompt
        'expected': '2-4 short mundane lines between the two, each prefixed with the full speaker name: orders of swords for '
                    'the Jarl/Legion, the Gray-Mane and Battle-Born feud, dragons, weather. No exposition ("As you know, I am '
                    'the blacksmith..."), no player involvement.',
        'expect_tools': [], 'forbid_tools': ['Attack', 'Brawl', 'Teleport'],
    },
    {
        'id': 'seed-08-adrianne-memory', 'type': 'single', 'npcs': ['Adrianne Avenicci'],
        'location': 'Warmaiden\'s', 'hour': 9, 'day': 19, 'weather': '',
        'equipment': {'Adrianne Avenicci': 'wears Blacksmith\'s Apron and Boots and uses Blacksmith\'s Hammer.'},
        'nearby': [],
        'summary': f'Adrianne Avenicci met {PLAYER} at Warmaiden\'s a week ago. {PLAYER} promised to bring her ten iron ingots '
                   'from the mine near Riverwood, and she promised a discount on a steel sword in return. She found him polite but '
                   'doubted he would follow through.',
        'history': [],
        'user': f'*{PLAYER} approaches Adrianne at the forge.* Morning, Adrianne.',
        'expected': 'Greets the player and brings up the promised iron ingots (or asks whether he brought them), consistent with the '
                    'memory. Her father is Proventus Avenicci. No tool needed.',
        'expect_tools': [], 'forbid_tools': ['Attack', 'Follow'],
    },
    {
        'id': 'seed-09-rolff-gray-quarter', 'type': 'single', 'npcs': ['Rolff Stone-Fist'],
        'location': 'Windhelm', 'hour': 22, 'day': 20, 'weather': 'Snow swirls through the cold night air.',
        'equipment': {'Rolff Stone-Fist': 'wears Clothes and Hat.'},
        'nearby': ['Suvaris Atheron'],
        'history': [],
        'user': 'What do you think of the Gray Quarter?',
        'expected': 'Drunk, sour, in-lore Nord bigotry toward the Dunmer of the Gray Quarter (no real-world slurs, no out-of-character '
                    'disclaimers). May challenge the player to a fistfight, but must NOT call Brawl before the player accepts.',
        'expect_tools': [], 'forbid_tools': ['Brawl', 'Attack'],
    },
    {
        'id': 'seed-10-anachronism-wifi', 'type': 'single', 'npcs': ['Hulda'],
        'location': 'The Bannered Mare', 'hour': 20, 'day': 12, 'weather': '',
        'equipment': {'Hulda': 'wears Barkeeper\'s Clothes.'},
        'nearby': ['Mikael', 'Saadia'],
        'history': [],
        'user': 'Do you have Wi-Fi in here?',
        'expected': 'In-world confusion ("Wye-fye? Never heard of it. Is that some Altmer drink?"), no acknowledgement of modern '
                    'technology, no meta talk. No tool needed.',
        'expect_tools': [], 'forbid_tools': ['Attack', 'Barter'],
    },
    {
        'id': 'seed-11-lydia-look', 'type': 'single', 'npcs': ['Lydia'],
        'location': 'Whiterun Stables', 'hour': 17, 'day': 14, 'weather': 'The sky is clear and the sun is bright.',
        'equipment': {'Lydia': 'wears Steel Plate Armor and uses Steel Sword and Iron Shield.'},
        'nearby': ['Skulvar Sable-Hilt'],
        'history': [],
        'user': 'Lydia, what do you see over there by the road?',
        'expected': 'Calls Look (to get a screenshot) rather than inventing what is there; a short line like "Let me see..." is fine.',
        'expect_tools': ['Look'], 'forbid_tools': ['Attack', 'TravelTo', 'Teleport'],
    },
    {
        'id': 'seed-12-guard-follow', 'type': 'single', 'npcs': ['Whiterun Guard'],
        'location': 'Whiterun', 'hour': 12, 'day': 12, 'weather': '',
        'equipment': {'Whiterun Guard': 'wears Whiterun Guard\'s Armor and uses Steel Sword.'},
        'nearby': [],
        'history': [],
        'user': 'Guard, follow me. I need a sword arm out on the road.',
        'expected': 'Refuses: his duty is to the Jarl and the city. No Follow, no LeadTo.',
        'expect_tools': [], 'forbid_tools': ['Follow', 'LeadTo', 'TravelTo'],
    },
]


def _time_group(hour: int) -> str:
    if hour <= 4:
        return 'at night'
    if hour <= 7:
        return 'in the early morning'
    if hour <= 11:
        return 'in the morning'
    if hour <= 14:
        return 'in the afternoon'
    if hour <= 19:
        return 'in the early evening'
    if hour <= 21:
        return 'in the late evening'
    return 'at night'


def _listing(names: list[str]) -> str:
    if not names:
        return ''
    if len(names) == 1:
        return names[0]
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def load_bios(repo_root: str = REPO_ROOT) -> dict[str, str]:
    path = os.path.join(repo_root, 'data', 'Skyrim', 'skyrim_characters.csv')
    bios: dict[str, str] = {}
    if not os.path.exists(path):
        return bios
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            name = (row.get('name') or '').strip()
            if name and name not in bios:
                bios[name] = (row.get('bio') or '').strip()
    return bios


def load_idle_names(repo_root: str = REPO_ROOT) -> list[str]:
    path = os.path.join(repo_root, 'data', 'Skyrim', 'skyrim_idles.csv')
    names: list[str] = []
    if not os.path.exists(path):
        return names
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if str(row.get('enabled', '')).strip().lower() == 'x' and row['idle_name'] not in names:
                names.append(row['idle_name'])
    return names


def load_prompts() -> dict[str, str]:
    from src.config.definitions.prompt_definitions import PromptDefinitions as P
    return {
        'single': P.get_skyrim_prompt_config_value().value,
        'multi': P.get_skyrim_multi_npc_prompt_config_value().value,
        'radiant': P.get_skyrim_radiant_prompt_config_value().value,
        'radiant_start': P.get_radiant_start_prompt_config_value().value,
    }


def build_tools(scenario_type: str, npcs: list[str], nearby: list[str], player: str | None,
                repo_root: str = REPO_ROOT, disabled: set[str] | None = None) -> list[dict[str, Any]]:
    """Mirror of FunctionManager.generate_context_aware_tools for a fixed scene."""
    flag = {'single': ('one_on_one', 'one-on-one'), 'multi': ('multi_npc', 'multi-npc'), 'radiant': ('radiant', 'radiant')}[scenario_type]
    idles = load_idle_names(repo_root)
    disabled = {d.lower() for d in (disabled or set())}
    entities = {
        'conversation': npcs,
        'conversation_w_player': npcs + ([player] if player else []),
        'nearby': nearby,
        'nearby_w_player': nearby + ([player] if player else []),
        'all_npcs': npcs + nearby,
        'all_npcs_w_player': npcs + nearby + ([player] if player else []),
    }
    tools = []
    for path in sorted(glob.glob(os.path.join(repo_root, 'data', 'actions', '*.json'))):
        with open(path, encoding='utf-8') as f:
            action = json.load(f)
        if action['name'].lower() in disabled or action.get('identifier', '').lower() in disabled:
            continue
        allowed = [g.lower().replace(' ', '') for g in action.get('allowed_games') or []]
        if allowed and 'skyrim' not in allowed:
            continue
        if not action.get(flag[0], action.get(flag[1], False)):
            continue
        tool = {'type': 'function', 'function': {'name': action['name'], 'description': action['description'],
                                                 'parameters': {'type': 'object', 'properties': {}}}}
        if 'parameters' in action:
            properties = deepcopy(action['parameters'])
            tool['function']['parameters']['properties'] = properties
            if 'required' in action:
                tool['function']['parameters']['required'] = action['required']
            skip = False
            scoped = with_entities = 0
            unscoped = False
            for param in properties.values():
                if param.get('enum_source') == 'idles':
                    if not idles:
                        skip = True
                    param['enum'] = idles
                    del param['enum_source']
                param.pop('resolve_to_id', None)
                scope = param.get('scope')
                if not scope:
                    unscoped = True
                    continue
                scoped += 1
                listing = _listing(entities.get(scope, []))
                if listing:
                    param['description'] = f"{param.get('description', '')} Available: {listing}".strip()
                    with_entities += 1
                else:
                    param['description'] = f"{param.get('description', '')} No entities available."
            if skip or (scoped and not with_entities and not unscoped):
                continue
        tools.append(tool)
    return tools


def build_seed_items(repo_root: str = REPO_ROOT, model: str = 'gpt-4.1-mini',
                     params: dict[str, Any] | None = None, disabled_actions: set[str] | None = None) -> list[dict[str, Any]]:
    bios = load_bios(repo_root)
    prompts = load_prompts()
    items = []
    for scenario in SCENARIOS:
        kind = scenario['type']
        npcs = scenario['npcs']
        player = None if kind == 'radiant' else PLAYER
        hour = scenario['hour']
        bio_for = lambda n: bios.get(n) or f'{n} is a resident of Skyrim.'
        equipment = ' '.join(f"{n} {scenario['equipment'][n]}" for n in npcs if n in scenario.get('equipment', {}))
        summary = scenario.get('summary', '')
        prompt_vars = {
            'name': npcs[-1], 'names': _listing(npcs), 'names_w_player': _listing(npcs + ([player] if player else [])),
            'player_name': player or '', 'player_description': '', 'player_equipment': PLAYER_EQUIPMENT if player else '',
            'equipment': equipment,
            'bio': bio_for(npcs[0]) if len(npcs) == 1 else '\n'.join(f'{n}: {bio_for(n)}' for n in npcs),
            'trust': 'a stranger',
            'conversation_summary': summary, 'location': scenario['location'],
            'time': hour - 12 if hour > 12 else hour, 'time_group': _time_group(hour), 'current_day': scenario['day'],
            'weather': scenario.get('weather', ''), 'actions': '', 'language': 'English',
        }
        prompt_vars['bios'] = prompt_vars['bio']
        prompt_vars['conversation_summaries'] = summary
        system = prompts[kind].format(**prompt_vars)
        messages: list[dict[str, Any]] = [{'role': 'system', 'content': system}]
        for role, text in scenario.get('history', []):
            messages.append({'role': role, 'content': text})
        if kind == 'radiant':
            user_text = prompts['radiant_start']
        elif kind == 'multi':
            user_text = f"{player}: {scenario['user']}"
        else:
            user_text = scenario['user']
        messages.append({'role': 'user', 'content': user_text})
        request_params = dict(params or DEFAULT_PARAMS)
        if kind != 'single':
            request_params['max_tokens'] = max(int(request_params.get('max_tokens', 0) or 0), MULTI_NPC_MIN_MAX_TOKENS)
        tools = build_tools(kind, npcs, scenario.get('nearby', []), player, repo_root, disabled_actions)
        items.append({
            'id': scenario['id'],
            'source': 'seed',
            'tags': {'scenario': kind, 'npcs': npcs, 'speaker': npcs[-1] if kind == 'single' else None,
                     'tools_offered': bool(tools), 'had_tool_call': None, 'followup_after_tool': False,
                     'had_image': False, 'expects_tool': bool(scenario['expect_tools'])},
            'request': {'model': model, 'messages': messages, 'tools': tools or None, 'params': request_params},
            'expected': {'notes': scenario['expected'], 'expect_tools': scenario['expect_tools'],
                         'forbid_tools': scenario['forbid_tools']},
            'prompt_vars': prompt_vars,
            'original_response': None,
        })
    return items
