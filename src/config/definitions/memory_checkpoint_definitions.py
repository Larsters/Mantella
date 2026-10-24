from src.config.types.config_value import ConfigValue, ConfigValueTag
from src.config.types.config_value_bool import ConfigValueBool
from src.config.types.config_value_int import ConfigValueInt


class MemoryCheckpointDefinitions:
    """Settings for crash-safe memory (src/memory_checkpoint): summaries and relationship updates during a conversation."""

    @staticmethod
    def get_memory_checkpoint_enabled_config_value() -> ConfigValue:
        description = """Save what was said during a conversation every few player turns and after a pause, instead of only when the conversation ends.
                        Without this, a game crash or a conversation that never ends properly means the NPC remembers nothing of it.
                        Each checkpoint summarises only the lines since the previous one (appended to the same summary file) and, with NPC Relationships enabled, updates the relationship record. The end of the conversation then only adds what is left, so nothing is counted twice.
                        If the Mantella server itself stops mid-conversation, the unsaved lines are kept in 'data/<game>/memory_checkpoints' and saved at the next start.
                        Cost: about one summary-model call per checkpoint (plus one relationship call if enabled). Requires 'Conversation Summaries' to be enabled."""
        return ConfigValueBool("memory_checkpoint_enabled", "Crash-Safe Memory Checkpoints", description, False, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_memory_checkpoint_every_n_turns_config_value() -> ConfigValue:
        description = """If Crash-Safe Memory Checkpoints are enabled, save a checkpoint after this many player turns since the last one. 0 = only on pauses."""
        return ConfigValueInt("memory_checkpoint_every_n_turns", "Checkpoint Every N Player Turns", description, 6, 0, 100, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_memory_checkpoint_idle_minutes_config_value() -> ConfigValue:
        description = """If Crash-Safe Memory Checkpoints are enabled, save a checkpoint when the player has not said anything for this many minutes (eg the game crashed). 0 = off."""
        return ConfigValueInt("memory_checkpoint_idle_minutes", "Checkpoint After Idle Minutes", description, 3, 0, 120, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_memory_checkpoint_recover_on_startup_config_value() -> ConfigValue:
        description = """If Crash-Safe Memory Checkpoints are enabled and the Mantella server stopped in the middle of a conversation, save that conversation's unsaved lines when the server starts again."""
        return ConfigValueBool("memory_checkpoint_recover_on_startup", "Recover Unfinished Conversations", description, True, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])
