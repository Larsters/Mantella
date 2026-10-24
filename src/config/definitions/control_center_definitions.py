from src.config.types.config_value import ConfigValue, ConfigValueTag
from src.config.types.config_value_bool import ConfigValueBool


class ControlCenterDefinitions:
    """Settings for the Control Center (/control)."""

    @staticmethod
    def get_control_center_persist_overrides_config_value() -> ConfigValue:
        description = """Keep the live changes made in the Control Center (http://localhost:4999/control) across Mantella restarts.
                        They are stored in 'data/control_center/live_overrides.json' in the Mantella user folder and loaded at startup. 'Reset' in the Control Center clears them (and the file).
                        Per-NPC overrides are never stored. If disabled, live changes last until Mantella is restarted."""
        return ConfigValueBool("control_center_persist_overrides", "Keep Control Center Changes", description, False, tags=[ConfigValueTag.advanced, ConfigValueTag.share_row])
