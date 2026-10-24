from src.config.types.config_value import ConfigValue, ConfigValueTag
from src.config.types.config_value_bool import ConfigValueBool
from src.config.types.config_value_float import ConfigValueFloat
from src.config.types.config_value_int import ConfigValueInt
from src.config.types.config_value_selection import ConfigValueSelection
from src.config.types.config_value_string import ConfigValueString


class BioGenerationDefinitions:
    """Settings for src/bio_generation (generated bios for generic NPCs). Off by default."""

    @staticmethod
    def get_bio_generation_enabled_config_value() -> ConfigValue:
        description = """Generate a short, in-lore bio with the LLM for NPCs that have no bio or only a weak / shared one (unnamed guards, bandits, mod NPCs), once per NPC and save it.
                        Saved under data/Skyrim/generated_bios/<world>/<NPC name> - <ref id>.json in your Mantella folder; edit or delete those files freely.
                        Costs one small LLM call per new NPC (uses the summary model if 'Custom Summary Model' is on, otherwise the main model)."""
        return ConfigValueBool("bio_generation_enabled", "Generated Bios", description, False, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_bio_generation_scope_config_value() -> ConfigValue:
        description = """Which NPCs get a generated bio.
                        - Missing only: NPCs not in the character list (they otherwise get 'You are a male Nord Bandit.').
                        - Missing and weak: also named NPCs whose bio is shorter than 'Generated Bios: Weak Bio Word Count'.
                        - Missing, weak and shared templates: also NPCs whose bio is a class template shared by everyone of that kind ('Whiterun Guard', 'Bandit'); the original text is kept and a description of this particular individual is appended."""
        options = ["Missing only", "Missing and weak", "Missing, weak and shared templates"]
        return ConfigValueSelection("bio_generation_scope", "Generated Bios: Scope", description, "Missing, weak and shared templates", options, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_bio_generation_min_words_config_value() -> ConfigValue:
        description = """Bios with fewer words than this count as weak and are expanded (when the scope includes weak bios)."""
        return ConfigValueInt("bio_generation_min_words", "Generated Bios: Weak Bio Word Count", description, 40, 0, 1000, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_bio_generation_wait_seconds_config_value() -> ConfigValue:
        description = """How long a conversation start may wait for a new bio (seconds). If generation takes longer, the conversation starts with the original bio and the generated one is used as soon as it is ready (from the next conversation at the latest).
                        0 = never wait (always generate in the background for next time)."""
        return ConfigValueFloat("bio_generation_wait_seconds", "Generated Bios: Max Wait", description, 6.0, 0.0, 30.0, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])

    @staticmethod
    def get_bio_generation_model_config_value() -> ConfigValue:
        description = """Model used for bio generation on the same service as the summary model (or the main model). Leave empty to use that model. A small, cheap model is enough."""
        return ConfigValueString("bio_generation_model", "Generated Bios: Model", description, "", tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])
