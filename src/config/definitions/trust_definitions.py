from src.config.types.config_value import ConfigValue, ConfigValueTag
from src.config.types.config_value_bool import ConfigValueBool


class TrustDefinitions:
    """Settings for trust and honesty (src/trust): trust-scaled compliance, checking claims, remembering lies."""

    @staticmethod
    def get_trust_and_honesty_enabled_config_value() -> ConfigValue:
        description = """NPCs do reasonable things you ask (look, search a barrel or chest with Loot, follow, wait, come over, give items) more readily the more they trust you, and do them with their actions instead of only talking.
                        When you claim something they could check (eg 'there is gold in that barrel'), a willing NPC checks with Look, Loot or MoveTo before acting on it. Claims about things that do not exist in Skyrim (bombs, grenades) are treated as nonsense, not crimes.
                        If you are caught lying, the NPC says so and doubts your word for the rest of the conversation. With NPC Relationships enabled, caught lies and a credibility score are also stored in the NPC's relationship record and shown in later conversations.
                        Adds a rule block to the 'living_npcs_v1' prompts (or in front of custom prompts), to the tool calling LLM prompt, and uses 'trust_description' for the Loot, Look, MoveTo, Inventory and ReportCrime actions."""
        return ConfigValueBool("trust_and_honesty_enabled", "Trust and Honesty", description, False, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_trust_tool_model_context_config_value() -> ConfigValue:
        description = """If Trust and Honesty and a Custom Tool Calling Model are enabled, tell the tool calling LLM who it is choosing actions for: the NPCs' names, the start of their bios, how well they know the player and their relationship record.
                        Without this, the tool calling LLM sees only the dialogue and cannot tell a guard from a housecarl or a friend from a stranger. Adds roughly 100-300 input tokens per tool call."""
        return ConfigValueBool("trust_tool_model_context", "Scene Context for Tool Calling LLM", description, True, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])
