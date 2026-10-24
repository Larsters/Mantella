from src.config.types.config_value import ConfigValue, ConfigValueTag
from src.config.types.config_value_bool import ConfigValueBool
from src.config.types.config_value_selection import ConfigValueSelection
from src.speech_flavor.presets import NONE_LABEL, preset_labels


class SpeechFlavorDefinitions:
    """Settings for src/speech_flavor (NPCs mix another language into their English). Off by default."""

    @staticmethod
    def get_speech_flavor_config_value() -> ConfigValue:
        description = """Give NPCs a speech flavour. 'German mix': NPCs still understand you perfectly and answer in clear English, but sprinkle in German words, German word order and false friends.
                        Fills the {speech_flavor} prompt variable, so the prompts must contain {speech_flavor} (e.g. in the Rules section) for this to have an effect.
                        Per-NPC: add "speech_flavor": "none" | "light" | "medium" | "heavy" | "german_mix:heavy" to the NPC's JSON in data/Skyrim/character_overrides."""
        return ConfigValueSelection("speech_flavor", "Speech Flavour", description, NONE_LABEL, preset_labels(), tags=[ConfigValueTag.share_row])

    @staticmethod
    def get_speech_flavor_intensity_config_value() -> ConfigValue:
        description = """How strong the speech flavour is.
                        - Light: about one foreign word per reply.
                        - Medium: one or two words or short phrases per reply, sometimes foreign word order.
                        - Heavy: about half of each sentence in the other language (key facts stay in English)."""
        return ConfigValueSelection("speech_flavor_intensity", "Speech Flavour: Intensity", description, "Medium", ["Light", "Medium", "Heavy"], tags=[ConfigValueTag.share_row])

    @staticmethod
    def get_speech_flavor_scope_config_value() -> ConfigValue:
        description = """Which NPCs get the speech flavour.
                        - By race: Nords at the chosen intensity; Imperials, Bretons, Redguards, Orcs and Elves one step lighter; Khajiit and Argonians keep their own way of speaking.
                        - All NPCs: everyone at the chosen intensity.
                        Per-NPC overrides in character_overrides always win."""
        return ConfigValueSelection("speech_flavor_scope", "Speech Flavour: Scope", description, "By race", ["By race", "All NPCs"], tags=[ConfigValueTag.share_row])

    @staticmethod
    def get_speech_flavor_tts_respell_config_value() -> ConfigValue:
        description = """Respell common foreign words for the English Piper voices (e.g. 'Scheiße' -> 'Shy-sa', 'Genau' -> 'Geh-now') so they sound closer to the real thing. Only the spoken audio changes; subtitles keep the proper spelling."""
        return ConfigValueBool("speech_flavor_tts_respell", "Speech Flavour: Respell for TTS", description, True, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])
