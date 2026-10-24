import json
from pathlib import Path

from src.actions.function_manager import FunctionManager
from src.config.config_loader import ConfigLoader
from src.conversation.context import Context

ACTIONS_DIR = Path(__file__).resolve().parents[2] / "data" / "actions"


def test_every_action_has_a_guided_description():
    for path in ACTIONS_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        guided = data.get("guided_description", "")
        assert guided, f"{path.name} has no guided_description"
        assert len(guided.split()) <= 50, f"{path.name}: keep tool descriptions lean"


def _tools(context: Context) -> dict[str, dict]:
    FunctionManager.load_all_actions([])
    return {t["function"]["name"]: t["function"] for t in FunctionManager.generate_context_aware_tools(context)}


def test_upstream_descriptions_by_default(default_config: ConfigLoader, default_context: Context):
    default_config.guided_action_descriptions = False
    tools = _tools(default_context)
    assert tools["Attack"]["description"] == "Make NPC(s) attack another NPC / the player."


def test_guided_descriptions_when_enabled(default_config: ConfigLoader, default_context: Context):
    default_config.guided_action_descriptions = True
    tools = _tools(default_context)
    assert "Never for insults alone" in tools["Attack"]["description"]
    assert "insulted again after a warning" in tools["Brawl"]["description"]


def test_internal_scope_field_not_sent_to_llm(default_context: Context):
    tools = _tools(default_context)
    for tool in tools.values():
        for param in tool["parameters"].get("properties", {}).values():
            assert "scope" not in param
    # ...while the entity list is still appended
    assert "Available:" in tools["Attack"]["parameters"]["properties"]["source"]["description"]
    # and the loaded definitions keep their scope for argument validation
    attack = next(a for a in FunctionManager._actions.values() if a["name"] == "Attack")
    assert attack["parameters"]["source"]["scope"] == "conversation"
