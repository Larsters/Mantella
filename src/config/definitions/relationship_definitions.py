from src.config.types.config_value import ConfigValue, ConfigValueTag
from src.config.types.config_value_bool import ConfigValueBool
from src.config.types.config_value_int import ConfigValueInt
from src.config.types.config_value_string import ConfigValueString
from src.config.definitions.prompt_definitions import PromptDefinitions
from src.relationships.updater import DEFAULT_UPDATE_PROMPT


class RelationshipDefinitions:
    """Settings for persistent per-NPC relationship records (opinion, facts, grudges, favours, promises)."""

    ALLOWED_UPDATE_PROMPT_VARIABLES = ["game", "language", "player_name", "names"]

    @staticmethod
    def get_relationship_tracking_enabled_config_value() -> ConfigValue:
        description = """Keep a structured record per NPC of what they think of the player: an opinion score (-100 to 100), facts they learned about you, grudges (with witnesses and the in-game day), favours/debts and promises.
                        The record is updated at the end of each conversation with one extra call to the summary LLM, and is stored next to the summaries as '<NPC name>_relationship.json' (you can view and edit it).
                        The record is added to the prompt via the '{relationship}' variable. If the prompt does not contain '{relationship}', it is added after the conversation summaries.
                        Requires 'Conversation Summaries' to be enabled."""
        return ConfigValueBool("relationship_tracking_enabled", "NPC Relationships", description, False, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_relationship_affects_trust_config_value() -> ConfigValue:
        description = """If NPC Relationships are enabled, refine the '{trust}' wording with the NPC's opinion of the player (eg 'a disliked acquaintance' instead of 'an acquaintance')."""
        return ConfigValueBool("relationship_affects_trust", "Relationship Affects Trust", description, True, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_relationship_token_budget_config_value() -> ConfigValue:
        description = """Maximum number of tokens used for the '{relationship}' text in the prompt (shared between all NPCs in the conversation). The opinion is always included; grudges, promises, debts and facts are added in that order while they fit."""
        return ConfigValueInt("relationship_token_budget", "Relationship Prompt Token Budget", description, 150, 40, 2000, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_relationship_opinion_decay_config_value() -> ConfigValue:
        description = """How many opinion points per in-game day an NPC's opinion drifts back towards neutral when you do not talk to them. 0 = never fades. Grudges are never forgotten and keep the opinion at least slightly negative until they are resolved."""
        return ConfigValueInt("relationship_opinion_decay_per_day", "Relationship Opinion Decay per Day", description, 0, 0, 100, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_relationship_update_prompt_config_value() -> ConfigValue:
        description = """The instructions sent to the summary LLM at the end of a conversation to update the NPC relationship records (only used if NPC Relationships are enabled). The reply must be JSON.
                        The following are dynamic variables that need to be contained in curly brackets {}:
                        game = the selected game
                        language = the selected language
                        player_name = the name of the player character
                        names = the names of the NPCs whose records are updated"""
        return ConfigValueString("relationship_update_prompt", "Relationship Update Prompt", description, DEFAULT_UPDATE_PROMPT,
                                 [PromptDefinitions.PromptChecker(RelationshipDefinitions.ALLOWED_UPDATE_PROMPT_VARIABLES)], tags=[ConfigValueTag.advanced])
