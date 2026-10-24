"""Eval-set transform for `build-set --trust`: make items look like what the server sends with
`prompt_profile = living_npcs_v1`, `guided_action_descriptions = True` and `trust_and_honesty_enabled = True` (W12).

- System prompt: seed items are re-rendered from the living_npcs_v1 templates (their prompt_vars); real turns
  already use that profile. Then W12's "Requests, claims and lies" block is inserted (one-on-one and multi only).
- Tools: real turns recorded with `custom_function_model = True` carry no tools (the main call had none), so they
  are rebuilt from data/actions for the NPCs in the conversation and the last "Characters nearby" list in the
  dialogue. Every description is what FunctionManager sends: trust_description > guided_description > description.
- `function_system`: the tool model's system prompt (function_llm_prompt + W12 "How to choose actions" + Scene),
  used by the `tool_model` turn mode.
"""
from __future__ import annotations

import glob
import json
import os
import re
from types import SimpleNamespace
from typing import Any

from src.evals import seeds
from src.evals.common import message_text

REPO_ROOT = seeds.REPO_ROOT
_NEARBY = re.compile(r'\(Characters nearby \(from nearest to furthest\): ([^)]*)\)')


class _Blank(dict):
    def __missing__(self, key: str) -> str:
        return ''


def _config() -> SimpleNamespace:
    return SimpleNamespace(trust_and_honesty_enabled=True, trust_tool_model_context=True, guided_action_descriptions=True)


def _actions_by_name(repo_root: str = REPO_ROOT) -> dict[str, dict[str, Any]]:
    actions = {}
    for path in glob.glob(os.path.join(repo_root, 'data', 'actions', '*.json')):
        with open(path, encoding='utf-8') as f:
            action = json.load(f)
        actions[action['name']] = action
    return actions


def last_nearby(messages: list[dict[str, Any]]) -> list[str]:
    for message in reversed(messages):
        if message.get('role') != 'user':
            continue
        found = _NEARBY.findall(message_text(message))
        if found:
            return [n.strip() for n in found[-1].split(',') if n.strip()]
    return []


def _bio_of(name: str, prompt_vars: dict[str, Any], npcs: list[str], csv_bios: dict[str, str]) -> str:
    if len(npcs) == 1 and prompt_vars.get('bio'):
        return str(prompt_vars['bio'])
    text = str(prompt_vars.get('bios') or prompt_vars.get('bio') or '')
    marker = f'{name}: '
    if marker in text:
        return text.split(marker, 1)[1][:400]
    return csv_bios.get(name, '')


def apply_trust(item: dict[str, Any], repo_root: str = REPO_ROOT, csv_bios: dict[str, str] | None = None) -> dict[str, Any]:
    from src import trust
    from src.trust import rules
    from src.prompt_profiles import get_profile_prompts
    from src.config.definitions.prompt_definitions import PromptDefinitions

    config = _config()
    csv_bios = seeds.load_bios(repo_root) if csv_bios is None else csv_bios
    scenario = item['tags'].get('scenario', 'single')
    prompt_vars = dict(item.get('prompt_vars') or {})
    request = item['request']
    messages = request['messages']
    npcs = list(item['tags'].get('npcs') or [])
    player = prompt_vars.get('player_name') or (None if scenario == 'radiant' else seeds.PLAYER)
    system_index = next((i for i, m in enumerate(messages) if m.get('role') == 'system'), None)

    if item.get('source') == 'seed':
        if str(prompt_vars.get('conversation_summary', '')).startswith('Relationship with the player'):
            prompt_vars['relationship'], prompt_vars['conversation_summary'] = prompt_vars['conversation_summary'], ''
            prompt_vars['conversation_summaries'] = ''
        profile = get_profile_prompts('living_npcs_v1')
        template = {'single': profile.prompt, 'multi': profile.multi_npc_prompt, 'radiant': profile.radiant_prompt}[scenario]
        messages[system_index]['content'] = template.format_map(_Blank(prompt_vars))
        if scenario == 'radiant':
            messages[-1]['content'] = profile.radiant_start_prompt
        item['prompt_vars'] = prompt_vars
    if system_index is not None and isinstance(messages[system_index].get('content'), str):
        if scenario == 'single':
            messages[system_index]['content'] = trust.insert_block(messages[system_index]['content'], rules.SINGLE_BLOCK, trust._SINGLE_MARKER)
        elif scenario == 'multi':
            messages[system_index]['content'] = trust.insert_block(messages[system_index]['content'], rules.MULTI_BLOCK, trust._MULTI_MARKER)

    if not request.get('tools'):
        nearby = [n for n in last_nearby(messages) if n not in npcs and n != player]
        request['tools'] = seeds.build_tools(scenario, npcs, nearby, player, repo_root) or None
        item['tags']['tools_rebuilt'] = True
    actions = _actions_by_name(repo_root)
    for tool in request.get('tools') or []:
        action = actions.get(tool['function']['name'])
        if action:
            tool['function']['description'] = trust.tool_description(action, config)
    item['tags']['tools_offered'] = bool(request.get('tools'))

    base = PromptDefinitions.get_function_llm_prompt_config_value().value.format(game='Skyrim')
    scene = trust.build_scene([SimpleNamespace(name=n, bio=_bio_of(n, prompt_vars, npcs, csv_bios)) for n in npcs],
                              player or '', str(prompt_vars.get('trust') or ''), str(prompt_vars.get('relationship') or ''))
    static = trust.function_prompt(config, base)
    item['function_system'] = f'{static}\n\n{scene}' if scene else static
    item['tags']['w12'] = True
    return item


def apply_annotations(items: list[dict[str, Any]], path: str) -> int:
    """`{item_id: {"notes", "expect_tools", "forbid_tools"}}` (hand labels for real turns). Returns how many matched."""
    with open(os.path.expanduser(path), encoding='utf-8') as f:
        labels = json.load(f)
    matched = 0
    for item in items:
        label = labels.get(item['id'])
        if label:
            item['expected'] = {**(item.get('expected') or {}), **label}
            item['tags']['expects_tool'] = bool(label.get('expect_tools'))
            matched += 1
    return matched
