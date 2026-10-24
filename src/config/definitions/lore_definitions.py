from src.config.types.config_value import ConfigValue, ConfigValueTag
from src.config.types.config_value_bool import ConfigValueBool
from src.config.types.config_value_int import ConfigValueInt


class LoreDefinitions:
    """Settings for the lore lookup feature (src/lore). Off by default = upstream behaviour."""

    @staticmethod
    def get_lore_enabled_config_value() -> ConfigValue:
        description = """Inject short lore notes (Skyrim only) about the current place, factions and the topic being discussed.
                        Scene notes fill the {lore} placeholder of the prompts (add {lore} to your prompt to use them).
                        Topic notes are attached to the player's message when 'Lore Per Turn' is enabled.
                        Notes ship in data/Skyrim/lore; add or override notes in <Mantella user folder>/data/Skyrim/lore/*.json."""
        return ConfigValueBool("lore_enabled", "Lore Lookup", description, False)

    @staticmethod
    def get_lore_per_turn_config_value() -> ConfigValue:
        description = """If enabled, notes about what the player just said are attached to that player message (each note at most once per conversation).
                        This works even if the prompt has no {lore} placeholder."""
        return ConfigValueBool("lore_per_turn", "Lore Per Turn", description, True, tags=[ConfigValueTag.advanced])

    @staticmethod
    def get_lore_max_notes_config_value() -> ConfigValue:
        description = """Maximum number of lore notes per injection (system prompt and each player turn separately)."""
        return ConfigValueInt("lore_max_notes", "Lore Max Notes", description, 3, 0, 10, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_lore_max_tokens_config_value() -> ConfigValue:
        description = """Hard token budget per injection (estimated at 4 characters per token). Notes that do not fit are skipped."""
        return ConfigValueInt("lore_max_tokens", "Lore Token Budget", description, 200, 0, 2000, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_lore_allow_spoilers_config_value() -> ConfigValue:
        description = """Allow notes marked as spoilers (quest twists, e.g. secret identities). Off by default because Mantella does not know your quest progress."""
        return ConfigValueBool("lore_allow_spoilers", "Lore Spoilers", description, False, tags=[ConfigValueTag.advanced])
