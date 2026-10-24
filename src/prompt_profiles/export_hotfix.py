"""Export a prompt profile for a Mantella server that does NOT have this feature.

Older servers know neither `prompt_profile`, `{knowledge_scope}` nor `guided_action_descriptions`.
This tool produces:
  1. config.ini lines for `skyrim_prompt` and `skyrim_multi_npc_prompt` (and radiant prompts) with
     the "What you know" section removed, encoded exactly like Mantella's ConfigFileWriter does
     (JSON string, '#' escaped as '\\#').
  2. Optionally, it rewrites `description` = `guided_description` in another server's data/actions/*.json
     (a .bak copy of each changed file is written first).

Usage (from the repo root):
  python -m src.prompt_profiles.export_hotfix                        # print config.ini lines
  python -m src.prompt_profiles.export_hotfix --actions-dir DIR      # dry run: show what would change
  python -m src.prompt_profiles.export_hotfix --actions-dir DIR --apply
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from src.prompt_profiles import get_profile_prompts

_KNOWLEDGE_SECTIONS = ("# What you know\n{knowledge_scope}\n\n", "# What they know\n{knowledge_scope}\n\n")
_KNOWLEDGE_REFERENCES = (", as described under What you know.", ", as described under What they know.")


def strip_knowledge_scope(prompt: str) -> str:
    """Removes the {knowledge_scope} section so the prompt works on servers without it."""
    for section in _KNOWLEDGE_SECTIONS:
        prompt = prompt.replace(section, "")
    for reference in _KNOWLEDGE_REFERENCES:
        prompt = prompt.replace(reference, ".")
    assert "{knowledge_scope}" not in prompt
    return prompt


def encode_ini_value(text: str) -> str:
    """Same encoding as ConfigFileWriter for multi-line strings."""
    return json.dumps(text).replace("#", "\\#")


def config_lines(profile_name: str = "living_npcs_v1", narration_handling=None) -> dict[str, str]:
    profile = get_profile_prompts(profile_name, narration_handling)
    if profile is None:
        raise SystemExit(f"Unknown profile {profile_name!r}")
    return {
        "skyrim_prompt": strip_knowledge_scope(profile.prompt),
        "skyrim_multi_npc_prompt": strip_knowledge_scope(profile.multi_npc_prompt),
        "skyrim_radiant_prompt": strip_knowledge_scope(profile.radiant_prompt),
        "radiant_start_prompt": profile.radiant_start_prompt,
        "radiant_end_prompt": profile.radiant_end_prompt,
    }


def patch_action_descriptions(source_dir: Path, target_dir: Path, apply: bool) -> list[str]:
    """Copies guided_description from source_dir's actions into target_dir's `description` fields."""
    changed = []
    for source in sorted(source_dir.glob("*.json")):
        target = target_dir / source.name
        if not target.exists():
            continue
        guided = json.loads(source.read_text(encoding="utf-8")).get("guided_description")
        if not guided:
            continue
        data = json.loads(target.read_text(encoding="utf-8"))
        if data.get("description") == guided:
            continue
        changed.append(f"{source.name}: {data.get('description')!r} -> {guided!r}")
        if apply:
            shutil.copy2(target, target.with_suffix(".json.bak"))
            text = target.read_text(encoding="utf-8")
            # Replace only the top-level description line to keep the file's formatting.
            new_text, count = re.subn(r'(?m)^(\s{1,4})"description": ".*?",\s*$',
                                      lambda m: f'{m.group(1)}"description": {json.dumps(guided, ensure_ascii=False)},',
                                      text, count=1)
            if count != 1 or json.loads(new_text).get("description") != guided:
                raise SystemExit(f"Could not patch {target}")
            target.write_text(new_text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="living_npcs_v1")
    parser.add_argument("--actions-dir", type=Path, help="data/actions folder of the server to patch")
    parser.add_argument("--apply", action="store_true", help="write the action changes (default: dry run)")
    args = parser.parse_args()

    if args.actions_dir:
        source_dir = Path(__file__).resolve().parents[2] / "data" / "actions"
        for line in patch_action_descriptions(source_dir, args.actions_dir, args.apply):
            print(line)
        print("APPLIED" if args.apply else "DRY RUN (use --apply)")
        return
    for key, value in config_lines(args.profile).items():
        print(f"{key} = {encode_ini_value(value)}")


if __name__ == "__main__":
    main()
