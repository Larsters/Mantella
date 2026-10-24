"""Live (in-memory) overrides for LLM and conversation settings, set from the Control Center (/control).

Changes apply to the NEXT LLM call without a restart. Nothing here touches config.ini unless
save_to_config() is called explicitly.

Merge order for an LLM request: config params < live overrides for the call type < per-NPC overrides
(dialogue only, matched by the speaking NPC's name). The result then goes through the model
capability adapter (src.llm.model_capabilities.adapt_request_params) if that module exists.

Thread-safe: the web server and the conversation threads read/write through one lock.
"""
from __future__ import annotations

import copy
import json
import threading
from typing import Any, Callable

import src.utils as utils

logger = utils.get_logger()

try:  # W3 (feat/openai-latest-models) provides per-model parameter filtering
    from src.llm.model_capabilities import adapt_request_params as _adapt_request_params
except Exception:  # pragma: no cover - depends on branch
    def _adapt_request_params(model: str, params: dict) -> dict:
        return params

LLM_CALL_TYPES = ('dialogue', 'memory')
LLM_KEYS = ('model', 'temperature', 'top_p', 'max_tokens', 'frequency_penalty', 'presence_penalty', 'reasoning_effort')
REASONING_EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh')
GENERAL_KEYS = ('max_response_sentences_single', 'max_response_sentences_multi', 'vision_enabled', 'disabled_actions')
NPC_KEYS = LLM_KEYS + ('max_response_sentences_single', 'max_response_sentences_multi')

# Clients that are not in usage_logger's call-type map (per-character ClientBase instances) count as dialogue
_CALL_TYPE_ALIASES = {'ClientBase': 'dialogue'}


class ValidationError(ValueError):
    pass


def _number(lo: float, hi: float, integer: bool = False) -> Callable[[Any], Any]:
    def check(v: Any) -> Any:
        if isinstance(v, bool):
            raise ValidationError('expected a number')
        try:
            n = int(v) if integer and float(v) == int(float(v)) else float(v)
        except (TypeError, ValueError):
            raise ValidationError('expected a number')
        if integer and not isinstance(n, int):
            raise ValidationError('expected a whole number')
        if not lo <= n <= hi:
            raise ValidationError(f'must be between {lo} and {hi}')
        return n
    return check


def _model(v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValidationError('expected a model name')
    return v.strip()


def _effort(v: Any) -> str:
    if v not in REASONING_EFFORTS:
        raise ValidationError(f'must be one of {", ".join(REASONING_EFFORTS)}')
    return v


def _bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise ValidationError('expected true/false')
    return v


def _str_list(v: Any) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValidationError('expected a list of names')
    return sorted(set(v))


VALIDATORS: dict[str, Callable[[Any], Any]] = {
    'model': _model,
    'temperature': _number(0, 2),
    'top_p': _number(0, 1),
    'max_tokens': _number(1, 128000, integer=True),
    'frequency_penalty': _number(-2, 2),
    'presence_penalty': _number(-2, 2),
    'reasoning_effort': _effort,
    'max_response_sentences_single': _number(1, 999, integer=True),
    'max_response_sentences_multi': _number(1, 999, integer=True),
    'vision_enabled': _bool,
    'disabled_actions': _str_list,
}


class RuntimeSettings:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._llm: dict[str, dict[str, Any]] = {}
        self._npc: dict[str, dict[str, Any]] = {}  # key: lower-cased NPC name
        self._npc_names: dict[str, str] = {}  # lower-cased -> display name
        self._general: dict[str, Any] = {}
        self._config: Any = None
        self._image_client: Any = None
        self.version = 0  # bumped on every change; lets the UI notice changes from elsewhere

    # ---------- setup ----------
    def attach_config(self, config: Any) -> None:
        with self._lock:
            self._config = config

    @property
    def config(self) -> Any:
        return self._config

    # ---------- writes ----------
    def _validated(self, values: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for key, value in values.items():
            if key not in allowed:
                errors[key] = 'unknown setting'
                continue
            if value is None or value == '':
                out[key] = None  # None = remove the override
                continue
            try:
                out[key] = VALIDATORS[key](value)
            except ValidationError as e:
                errors[key] = str(e)
        if errors:
            raise ValidationError(json.dumps(errors))
        return out

    @staticmethod
    def _merge_into(target: dict[str, Any], values: dict[str, Any]) -> None:
        for key, value in values.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value

    def set_llm(self, call_type: str, values: dict[str, Any]) -> None:
        if call_type not in LLM_CALL_TYPES:
            raise ValidationError(f'unknown call type {call_type!r}')
        clean = self._validated(values, LLM_KEYS)
        with self._lock:
            self._merge_into(self._llm.setdefault(call_type, {}), clean)
            self.version += 1
        logger.info(f'Control Center: live {call_type} overrides now {self._llm.get(call_type)}')

    def set_npc(self, npc_name: str, values: dict[str, Any]) -> None:
        if not isinstance(npc_name, str) or not npc_name.strip():
            raise ValidationError('NPC name required')
        clean = self._validated(values, NPC_KEYS)
        key = npc_name.strip().lower()
        with self._lock:
            self._npc_names[key] = npc_name.strip()
            self._merge_into(self._npc.setdefault(key, {}), clean)
            if not self._npc[key]:
                del self._npc[key]
                del self._npc_names[key]
            self.version += 1
        logger.info(f'Control Center: live overrides for {npc_name} now {self._npc.get(key)}')

    def set_general(self, values: dict[str, Any]) -> None:
        clean = self._validated(values, GENERAL_KEYS)
        with self._lock:
            old_disabled = self._general.get('disabled_actions')
            self._merge_into(self._general, clean)
            self.version += 1
            actions_changed = 'disabled_actions' in clean and self._general.get('disabled_actions') != old_disabled
        logger.info(f'Control Center: live general overrides now {self._general}')
        if actions_changed:
            self._reload_actions()

    def reset(self, scope: str = 'all', npc_name: str | None = None) -> None:
        with self._lock:
            had_actions = 'disabled_actions' in self._general
            if scope == 'all':
                self._llm.clear(); self._npc.clear(); self._npc_names.clear(); self._general.clear()
            elif scope in LLM_CALL_TYPES:
                self._llm.pop(scope, None)
            elif scope == 'general':
                self._general.clear()
            elif scope == 'npc':
                if npc_name:
                    key = npc_name.strip().lower()
                    self._npc.pop(key, None); self._npc_names.pop(key, None)
                else:
                    self._npc.clear(); self._npc_names.clear()
            else:
                raise ValidationError(f'unknown scope {scope!r}')
            self.version += 1
            actions_changed = had_actions and 'disabled_actions' not in self._general
        if actions_changed:
            self._reload_actions()

    # ---------- reads used by the hooks ----------
    @staticmethod
    def _current_speaker() -> str | None:
        try:
            import src.usage_logger as usage_logger
            speaker = usage_logger._context.get('speaker')
            return speaker if isinstance(speaker, str) else None
        except Exception:
            return None

    def llm_overrides(self, call_type: str, npc_name: str | None = None) -> dict[str, Any]:
        """Live LLM params for a call type (may include 'model'): live call-type values < per-NPC values.

        call_type is the usage_logger name ('dialogue', 'memory', ...). Per-character clients
        ('ClientBase') are dialogue, but keep their own model unless a per-NPC model is set.
        """
        per_character_client = call_type in _CALL_TYPE_ALIASES
        call_type = _CALL_TYPE_ALIASES.get(call_type, call_type)
        with self._lock:
            result = dict(self._llm.get(call_type, {}))
            if per_character_client:
                result.pop('model', None)
            if call_type == 'dialogue':
                name = npc_name or self._current_speaker()
                if name:
                    result.update({k: v for k, v in self._npc.get(name.lower(), {}).items() if k in LLM_KEYS})
        return result

    def resolve_request(self, call_type: str, model: str, params: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        """config params < live < per-NPC. Returns (model, new params dict). Never raises."""
        merged: dict[str, Any] = dict(params or {})
        try:
            overrides = self.llm_overrides(call_type)
            model = overrides.pop('model', None) or model
            merged.update(overrides)
        except Exception as e:
            logger.debug(f'Runtime settings merge failed: {e}')
        return model, merged

    @staticmethod
    def adapt(model: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return _adapt_request_params(model, params)
        except Exception as e:
            logger.debug(f'Model capability adapter failed: {e}')
            return params

    def max_response_sentences(self, is_multi_npc: bool, default: int, npc_name: str | None = None) -> int:
        key = 'max_response_sentences_multi' if is_multi_npc else 'max_response_sentences_single'
        with self._lock:
            value = self._general.get(key, default)
            if npc_name and not is_multi_npc:
                value = self._npc.get(npc_name.lower(), {}).get(key, value)
        return value

    def vision_override(self) -> bool | None:
        with self._lock:
            return self._general.get('vision_enabled')

    def prepare_vision(self, client: Any) -> bool | None:
        """Hook for ClientBase._should_enable_vision. Returns False to force vision off, None to let
        the client decide. When vision is forced on and the client has no image client, one is
        created lazily from the config."""
        live = self.vision_override()
        if live is None:
            return None
        if not live:
            return False
        try:
            if getattr(client, '_image_client', None) is None:
                image_client = self._get_image_client()
                if image_client is None:
                    return None
                client._image_client = image_client
            if getattr(client, '_vision_mode', None) is not None and client._vision_mode.value == 'disabled':
                client._vision_mode = client._determine_vision_mode()
        except Exception as e:
            logger.warning(f'Could not enable vision live: {e}')
        return None

    def _get_image_client(self) -> Any:
        with self._lock:
            if self._image_client is None and self._config is not None:
                from src.llm.image_client import ImageClient
                logger.info('Control Center: setting up vision language model...')
                self._image_client = ImageClient(self._config)
            return self._image_client

    def disabled_actions(self) -> list[str] | None:
        with self._lock:
            value = self._general.get('disabled_actions')
            return list(value) if value is not None else None

    def _reload_actions(self) -> None:
        try:
            from src.actions.function_manager import FunctionManager
            disabled = self.disabled_actions()
            if disabled is None:
                disabled = list(getattr(self._config, 'disabled_actions', []) or [])
            FunctionManager.load_all_actions(disabled_actions=disabled)
        except Exception as e:
            logger.error(f'Control Center: could not reload actions: {e}')

    # ---------- state for the UI ----------
    def config_values(self) -> dict[str, Any]:
        c = self._config
        if c is None:
            return {}
        llm_params = dict(getattr(c, 'llm_params', None) or {})
        summary_params = dict(getattr(c, 'summary_llm_params', None) or {})
        return {
            'dialogue': {'model': getattr(c, 'llm', None), **{k: llm_params[k] for k in LLM_KEYS if k in llm_params}},
            'memory': {'model': getattr(c, 'summary_llm', None), **{k: summary_params[k] for k in LLM_KEYS if k in summary_params}},
            'general': {
                'max_response_sentences_single': getattr(c, 'max_response_sentences_single', None),
                'max_response_sentences_multi': getattr(c, 'max_response_sentences_multi', None),
                'vision_enabled': getattr(c, 'vision_enabled', None),
                'disabled_actions': sorted(getattr(c, 'disabled_actions', []) or []),
            },
            'llm_api': getattr(c, 'llm_api', None),
            'summary_llm_api': getattr(c, 'summary_llm_api', None),
            'advanced_actions_enabled': getattr(c, 'advanced_actions_enabled', None),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            live = {
                'llm': copy.deepcopy(self._llm),
                'general': copy.deepcopy(self._general),
                'npc': {self._npc_names[k]: dict(v) for k, v in self._npc.items()},
            }
            version = self.version
        config = self.config_values()
        effective: dict[str, Any] = {}
        for call_type in LLM_CALL_TYPES:
            effective[call_type] = {**config.get(call_type, {}), **live['llm'].get(call_type, {})}
        effective['general'] = {**config.get('general', {}), **live['general']}
        return {'version': version, 'config': config, 'live': live, 'effective': effective}

    # ---------- save ----------
    def save_to_config(self) -> list[str]:
        """Write live dialogue/memory/general values into config.ini through Mantella's own config
        definitions (the same path the Gradio UI uses), then clear those live overrides.
        Per-NPC overrides stay live-only. Returns the config keys written."""
        c = self._config
        if c is None:
            raise RuntimeError('No config attached')
        definitions = c.definitions
        with self._lock:
            llm = copy.deepcopy(self._llm)
            general = copy.deepcopy(self._general)

        changes: list[tuple[str, Any]] = []
        for call_type, model_key, params_key in (('dialogue', 'model', 'llm_params'), ('memory', 'summary_llm', 'summary_llm_params')):
            values = dict(llm.get(call_type, {}))
            if not values:
                continue
            if 'model' in values:
                changes.append((model_key, values.pop('model')))
            if values:
                current = definitions.get_config_value_definition(params_key).value
                try:
                    params = json.loads(current.replace('\n', '')) if current and current.strip() else {}
                except Exception:
                    raise ValidationError(f'{params_key} in config.ini is not valid JSON; fix it in the Mantella UI first')
                params.update(values)
                changes.append((params_key, json.dumps(params, indent=4)))
        for key in ('max_response_sentences_single', 'max_response_sentences_multi', 'vision_enabled', 'disabled_actions'):
            if key in general:
                changes.append((key, general[key]))

        # validate everything before writing anything
        for key, value in changes:
            result = definitions.get_config_value_definition(key).does_value_cause_error(value)
            if not result.is_success:
                raise ValidationError(f'{key}: {result.error_message}')
        written = []
        for key, value in changes:
            definition = definitions.get_config_value_definition(key)
            if definition.value != value:
                definition.value = value  # triggers ConfigLoader's writer (whole file, from the shared definitions)
                written.append(key)
        with self._lock:
            for call_type in ('dialogue', 'memory'):
                self._llm.pop(call_type, None)
            self._general.clear()
            self.version += 1
        if written:
            logger.log(24, f'Control Center saved to config.ini: {", ".join(written)}')
        return written


_settings = RuntimeSettings()


def get() -> RuntimeSettings:
    return _settings


# Module-level shortcuts used by the hooks in existing files
def llm_overrides(call_type: str, npc_name: str | None = None) -> dict[str, Any]:
    return _settings.llm_overrides(call_type, npc_name)


def resolve_request(call_type: str, model: str, params: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    return _settings.resolve_request(call_type, model, params)


def adapt(model: str, params: dict[str, Any]) -> dict[str, Any]:
    return _settings.adapt(model, params)


def max_response_sentences(is_multi_npc: bool, default: int, npc_name: str | None = None) -> int:
    return _settings.max_response_sentences(is_multi_npc, default, npc_name)


def prepare_vision(client: Any) -> bool | None:
    return _settings.prepare_vision(client)
