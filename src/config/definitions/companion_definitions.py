from src.config.types.config_value import ConfigValue
from src.config.types.config_value_bool import ConfigValueBool
from src.config.types.config_value_string import ConfigValueString


class CompanionDefinitions:
    """Settings for the opt-in self-aware companion (see src/companion_profile.py)."""

    @staticmethod
    def get_companion_enabled_config_value() -> ConfigValue:
        description = """If enabled, the companion character(s) below get their own one-on-one prompt (data/<game>/companion/<name>_prompt.txt),
                        the contents of about_me.txt (facts about the real player) via the {about_player} placeholder, and the RememberAboutPlayer tool.
                        about_me.txt is never put into multi-NPC or radiant prompts."""
        return ConfigValueBool("companion_enabled", "Self-Aware Companion", description, False)

    @staticmethod
    def get_companion_names_config_value() -> ConfigValue:
        description = """Comma-separated names of the companion character(s), eg 'Lydia'."""
        return ConfigValueString("companion_names", "Companion Names", description, "Lydia")

    @staticmethod
    def get_companion_about_me_path_config_value() -> ConfigValue:
        description = """Path of the about_me.txt file. Leave empty to use about_me.txt in the Mantella user folder (next to config.ini)."""
        return ConfigValueString("companion_about_me_path", "about_me.txt Path", description, "")

    @staticmethod
    def get_companion_group_memories_config_value() -> ConfigValue:
        description = """If disabled (default), the companion's memory summaries are left out of shared multi-NPC / radiant prompts,
                        because they can contain what the player told her in private. Enable to let her remember past conversations in groups."""
        return ConfigValueBool("companion_group_memories", "Companion Memories in Groups", description, False)
