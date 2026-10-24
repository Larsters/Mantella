"""Tests for src/runtime_settings.py (Control Center live overrides) and its ClientBase hooks."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.runtime_settings as runtime_settings
import src.usage_logger as usage_logger
from src.llm.client_base import ClientBase
from src.llm.messages import UserMessage


@pytest.fixture
def rs(monkeypatch):
    """A fresh RuntimeSettings installed as the module singleton."""
    fresh = runtime_settings.RuntimeSettings()
    monkeypatch.setattr(runtime_settings, '_settings', fresh)
    monkeypatch.setattr(usage_logger, '_context', {})
    return fresh


@pytest.fixture(autouse=True)
def no_usage_file(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_logger, '_log_path', str(tmp_path / 'usage.jsonl'))


# ---------- merge order ----------

def test_no_overrides_returns_config_params_unchanged(rs):
    model, params = rs.resolve_request('dialogue', 'gpt-4.1-mini', {'max_tokens': 250, 'temperature': 0.7})
    assert model == 'gpt-4.1-mini'
    assert params == {'max_tokens': 250, 'temperature': 0.7}


def test_live_overrides_config(rs):
    rs.set_llm('dialogue', {'temperature': 1.2, 'model': 'gpt-5-mini'})
    base = {'max_tokens': 250, 'temperature': 0.7}
    model, params = rs.resolve_request('dialogue', 'gpt-4.1-mini', base)
    assert model == 'gpt-5-mini'
    assert params == {'max_tokens': 250, 'temperature': 1.2}
    assert base == {'max_tokens': 250, 'temperature': 0.7}, 'config params must not be mutated'


def test_per_npc_overrides_live_and_config(rs):
    rs.set_llm('dialogue', {'temperature': 1.2, 'top_p': 0.9})
    rs.set_npc('Lydia', {'temperature': 0.3, 'model': 'gpt-5.4-mini'})
    usage_logger.set_context(speaker='lydia')  # case-insensitive match
    model, params = rs.resolve_request('dialogue', 'gpt-4.1-mini', {'temperature': 0.7, 'max_tokens': 250})
    assert model == 'gpt-5.4-mini'
    assert params == {'temperature': 0.3, 'top_p': 0.9, 'max_tokens': 250}

    usage_logger.set_context(speaker='Ulfric Stormcloak')
    model, params = rs.resolve_request('dialogue', 'gpt-4.1-mini', {'temperature': 0.7})
    assert model == 'gpt-4.1-mini'
    assert params['temperature'] == 1.2


def test_overrides_are_scoped_by_call_type(rs):
    rs.set_llm('dialogue', {'temperature': 1.5})
    rs.set_npc('Lydia', {'temperature': 0.1})
    usage_logger.set_context(speaker='Lydia')
    _, params = rs.resolve_request('memory', 'gpt-4.1-mini', {'temperature': 0.7})
    assert params == {'temperature': 0.7}, 'dialogue and per-NPC overrides must not touch memory summaries'
    rs.set_llm('memory', {'max_tokens': 900})
    _, params = rs.resolve_request('memory', 'gpt-4.1-mini', {})
    assert params == {'max_tokens': 900}


def test_per_character_client_keeps_its_own_model(rs):
    rs.set_llm('dialogue', {'model': 'gpt-5-mini', 'temperature': 0.5})
    model, params = rs.resolve_request('ClientBase', 'claude-opus', {})
    assert model == 'claude-opus'
    assert params == {'temperature': 0.5}


def test_none_removes_override_and_reset(rs):
    rs.set_llm('dialogue', {'temperature': 1.2, 'top_p': 0.5})
    rs.set_llm('dialogue', {'temperature': None})
    assert rs.llm_overrides('dialogue') == {'top_p': 0.5}
    rs.set_npc('Lydia', {'temperature': 0.2})
    rs.reset('dialogue')
    assert rs.llm_overrides('dialogue') == {}
    assert rs.snapshot()['live']['npc'] == {'Lydia': {'temperature': 0.2}}
    rs.reset('npc', 'lydia')
    assert rs.snapshot()['live']['npc'] == {}


@pytest.mark.parametrize('values', [
    {'temperature': 3}, {'top_p': -0.1}, {'max_tokens': 2.5}, {'reasoning_effort': 'turbo'},
    {'model': ''}, {'stop': ['#']}, {'temperature': 'hot'},
])
def test_validation_rejects_bad_values(rs, values):
    if values == {'model': ''}:
        rs.set_llm('dialogue', values)  # empty string means "remove"
        assert rs.llm_overrides('dialogue') == {}
        return
    with pytest.raises(runtime_settings.ValidationError):
        rs.set_llm('dialogue', values)
    assert rs.llm_overrides('dialogue') == {}


def test_max_response_sentences(rs):
    assert rs.max_response_sentences(False, 4, 'Lydia') == 4
    rs.set_general({'max_response_sentences_single': 2, 'max_response_sentences_multi': 6})
    assert rs.max_response_sentences(False, 4, 'Lydia') == 2
    assert rs.max_response_sentences(True, 12, 'Lydia') == 6
    rs.set_npc('Lydia', {'max_response_sentences_single': 1})
    assert rs.max_response_sentences(False, 4, 'Lydia') == 1
    assert rs.max_response_sentences(False, 4, 'Nazeem') == 2


def test_adapter_is_applied(rs, monkeypatch):
    monkeypatch.setattr(runtime_settings, '_adapt_request_params',
                        lambda model, params: {('max_completion_tokens' if k == 'max_tokens' else k): v for k, v in params.items()})
    assert rs.adapt('gpt-5-mini', {'max_tokens': 100}) == {'max_completion_tokens': 100}


def test_vision_override(rs):
    client = SimpleNamespace(_image_client=object(), _vision_mode=None)
    assert rs.prepare_vision(client) is None
    rs.set_general({'vision_enabled': False})
    assert rs.prepare_vision(client) is False
    rs.set_general({'vision_enabled': None})
    assert rs.prepare_vision(client) is None


def test_disabled_actions_reload_function_manager(rs, monkeypatch):
    calls = []
    from src.actions.function_manager import FunctionManager
    monkeypatch.setattr(FunctionManager, 'load_all_actions', staticmethod(lambda disabled_actions=None: calls.append(disabled_actions)))
    rs.attach_config(SimpleNamespace(disabled_actions=['Brawl']))
    rs.set_general({'disabled_actions': ['Teleport', 'Loot']})
    assert calls[-1] == ['Loot', 'Teleport']
    rs.reset('general')
    assert calls[-1] == ['Brawl'], 'reset goes back to the config list'


# ---------- ClientBase hooks: apply-live affects the NEXT call ----------

def _local_client(params=None):
    return ClientBase('http://127.0.0.1:9/v1', 'gpt-4.1-mini', params if params is not None else {'max_tokens': 250, 'temperature': 0.7}, 4096)


def _fake_sync_client(calls):
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Hi'))],
                                 usage={'prompt_tokens': 10, 'completion_tokens': 2})
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kw: calls.append(kw) or completion
    return client


def test_request_call_uses_live_overrides_on_next_call(rs):
    client = _local_client()
    calls = []
    client.generate_sync_client = lambda: _fake_sync_client(calls)
    msg = UserMessage(None, 'hello')  # config param not needed for get_openai_message

    client.request_call(msg)
    assert calls[-1]['model'] == 'gpt-4.1-mini' and calls[-1]['temperature'] == 0.7

    # SummaryLLMClient maps to 'memory'; a bare ClientBase maps to 'dialogue' (per-character client)
    rs.set_llm('dialogue', {'temperature': 1.3, 'top_p': 0.8, 'model': 'gpt-5-mini'})
    client.request_call(msg)
    assert calls[-1]['temperature'] == 1.3 and calls[-1]['top_p'] == 0.8
    assert calls[-1]['model'] == 'gpt-4.1-mini', 'per-character clients keep their model'
    assert client._request_params == {'max_tokens': 250, 'temperature': 0.7}, 'client config untouched'


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _fake_async_client(calls):
    chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content='Hello.', tool_calls=None))])
    usage_chunk = SimpleNamespace(usage={'prompt_tokens': 5, 'completion_tokens': 1}, choices=[])

    async def create(**kw):
        calls.append(kw)
        return _FakeStream([chunk, usage_chunk])

    async def close():
        pass
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)


def _stream(client, is_multi_npc=False):
    async def run():
        return [item async for item in client.streaming_call(UserMessage(None, 'hello'), is_multi_npc=is_multi_npc)]
    return asyncio.run(run())


def test_streaming_call_dialogue_model_and_params_change_live(rs, monkeypatch):
    client = _local_client()
    client.__class__ = type('LLMClient', (ClientBase,), {})  # usage_logger maps by class name -> 'dialogue'
    calls = []
    client.generate_async_client = lambda: _fake_async_client(calls)

    assert ('content', 'Hello.') in _stream(client)
    assert calls[-1]['model'] == 'gpt-4.1-mini' and calls[-1]['temperature'] == 0.7

    rs.set_llm('dialogue', {'model': 'gpt-5-mini', 'temperature': 0.2, 'reasoning_effort': 'low'})
    _stream(client)
    assert calls[-1]['model'] == 'gpt-5-mini'
    assert calls[-1]['temperature'] == 0.2 and calls[-1]['reasoning_effort'] == 'low'

    rs.set_npc('Lydia', {'max_tokens': 80})
    usage_logger.set_context(speaker='Lydia')
    _stream(client)
    assert calls[-1]['max_tokens'] == 80

    _stream(client, is_multi_npc=True)
    assert calls[-1]['max_tokens'] == 250, 'multi-NPC floor of 250 still applies after overrides'

    rs.reset('all')
    _stream(client)
    assert calls[-1]['model'] == 'gpt-4.1-mini' and 'reasoning_effort' not in calls[-1]

    # usage log records the model actually used
    lines = [json.loads(l) for l in open(usage_logger._log_path, encoding='utf-8')]
    assert [l['model'] for l in lines][:2] == ['gpt-4.1-mini', 'gpt-5-mini']


def test_streaming_call_passes_through_adapter(rs, monkeypatch):
    # play integration: ClientBase calls W3's model_capabilities.adapt_request_params once, with the overridden model
    from src.llm import client_base as client_base_module
    monkeypatch.setattr(client_base_module.model_capabilities, 'adapt_request_params',
                        lambda model, params, base_url=None: {k: v for k, v in params.items() if k != 'temperature'} if model.startswith('gpt-5') else params)
    client = _local_client()
    calls = []
    client.generate_async_client = lambda: _fake_async_client(calls)
    rs.set_llm('dialogue', {'model': 'gpt-5-mini'})
    client.__class__ = type('LLMClient', (ClientBase,), {})
    _stream(client)
    assert calls[-1]['model'] == 'gpt-5-mini' and 'temperature' not in calls[-1]


# ---------- save to config.ini ----------

def test_save_to_config_writes_through_definitions(rs, tmp_path):
    from src.config.config_loader import ConfigLoader
    from src.config.definitions.game_definitions import GameEnum
    config = ConfigLoader(mygame_folder_path=str(tmp_path), game_override=GameEnum.SKYRIM)
    rs.attach_config(config)
    rs.set_llm('dialogue', {'temperature': 0.9, 'model': 'gpt-5-mini'})
    rs.set_general({'max_response_sentences_single': 3, 'vision_enabled': True})
    rs.set_npc('Lydia', {'temperature': 0.2})

    written = rs.save_to_config()
    assert set(written) == {'model', 'llm_params', 'max_response_sentences_single', 'vision_enabled'}
    text = (tmp_path / 'config.ini').read_text(encoding='utf-8')
    assert 'gpt-5-mini' in text and '0.9' in text and 'max_response_sentences_single = 3' in text
    assert config.has_any_config_value_changed  # Mantella reloads on the next request
    config.update_config_loader_with_changed_config_values()
    assert config.llm == 'gpt-5-mini'
    assert config.llm_params['temperature'] == 0.9 and config.llm_params['max_tokens'] == 250  # existing params kept
    assert config.max_response_sentences_single == 3

    snap = rs.snapshot()
    assert snap['live']['llm'] == {} and snap['live']['general'] == {}
    assert snap['live']['npc'] == {'Lydia': {'temperature': 0.2}}, 'per-NPC stays live-only'


# ---------- persistence across restarts (control_center_persist_overrides) ----------

def _persist_config(tmp_path, enabled=True, **extra):
    return SimpleNamespace(save_folder=str(tmp_path), control_center_persist_overrides=enabled, disabled_actions=['Brawl'], **extra)


def _no_action_reload(monkeypatch):
    calls = []
    from src.actions.function_manager import FunctionManager
    monkeypatch.setattr(FunctionManager, 'load_all_actions', staticmethod(lambda disabled_actions=None: calls.append(disabled_actions)))
    return calls


def test_persisted_overrides_roundtrip_through_a_restart(rs, tmp_path, monkeypatch):
    calls = _no_action_reload(monkeypatch)
    rs.attach_config(_persist_config(tmp_path))
    rs.set_llm('dialogue', {'model': 'gpt-5.6-luna', 'temperature': 0.8})
    rs.set_llm('memory', {'max_tokens': 400})
    rs.set_general({'max_response_sentences_single': 2, 'disabled_actions': ['Loot', 'Teleport']})
    rs.set_npc('Lydia', {'temperature': 0.1})
    path = tmp_path / 'data' / 'control_center' / 'live_overrides.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['llm'] == {'dialogue': {'model': 'gpt-5.6-luna', 'temperature': 0.8}, 'memory': {'max_tokens': 400}}
    assert saved['general'] == {'max_response_sentences_single': 2, 'disabled_actions': ['Loot', 'Teleport']}
    assert 'npc' not in saved and 'Lydia' not in path.read_text(), 'per-NPC overrides are never persisted'
    assert rs.snapshot()['persist']['enabled'] is True and rs.snapshot()['persist']['file_exists'] is True

    # "Restart": a brand-new RuntimeSettings attached to the same user folder
    calls.clear()
    restarted = runtime_settings.RuntimeSettings()
    restarted.attach_config(_persist_config(tmp_path))
    model, params = restarted.resolve_request('dialogue', 'gpt-4.1-mini', {'temperature': 0.5, 'max_tokens': 250})
    assert model == 'gpt-5.6-luna' and params == {'temperature': 0.8, 'max_tokens': 250}
    assert restarted.resolve_request('memory', 'gpt-4.1-mini', {})[1] == {'max_tokens': 400}
    assert restarted.max_response_sentences(False, 4) == 2
    assert calls == [['Loot', 'Teleport']], 'restored action list is applied at startup'
    assert restarted.snapshot()['live']['npc'] == {}
    assert restarted.snapshot()['persist']['restored_from'] == str(path)

    # Reset to config clears the file
    restarted.reset('all')
    assert not path.exists()
    assert calls[-1] == ['Brawl']


def test_persistence_off_by_default_and_when_disabled(rs, tmp_path, monkeypatch):
    _no_action_reload(monkeypatch)
    rs.attach_config(_persist_config(tmp_path, enabled=False))
    rs.set_llm('dialogue', {'temperature': 0.8})
    assert not (tmp_path / 'data').exists()
    assert rs.snapshot()['persist']['enabled'] is False
    # A file left from an earlier session is ignored while the setting is off
    path = tmp_path / 'data' / 'control_center' / 'live_overrides.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'llm': {'dialogue': {'temperature': 1.5}}}), encoding='utf-8')
    fresh = runtime_settings.RuntimeSettings()
    fresh.attach_config(_persist_config(tmp_path, enabled=False))
    assert fresh.llm_overrides('dialogue') == {}


def test_persisted_file_with_bad_values_is_loaded_leniently(rs, tmp_path, monkeypatch):
    _no_action_reload(monkeypatch)
    path = tmp_path / 'data' / 'control_center' / 'live_overrides.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'llm': {'dialogue': {'temperature': 9, 'top_p': 0.5, 'bogus': 1}, 'radiant': {'temperature': 1}},
                                'general': {'vision_enabled': 'yes', 'max_response_sentences_multi': 3}}), encoding='utf-8')
    rs.attach_config(_persist_config(tmp_path))
    assert rs.llm_overrides('dialogue') == {'top_p': 0.5}
    assert rs.snapshot()['live']['general'] == {'max_response_sentences_multi': 3}
    path.write_text('{broken', encoding='utf-8')
    fresh = runtime_settings.RuntimeSettings()
    fresh.attach_config(_persist_config(tmp_path))
    assert fresh.llm_overrides('dialogue') == {} and fresh.snapshot()['persist']['error']


def test_save_to_config_clears_persisted_file(rs, tmp_path, monkeypatch):
    from src.config.config_loader import ConfigLoader
    from src.config.definitions.game_definitions import GameEnum
    config = ConfigLoader(mygame_folder_path=str(tmp_path), game_override=GameEnum.SKYRIM)
    assert config.control_center_persist_overrides is False  # upstream default
    config.control_center_persist_overrides = True
    rs.attach_config(config)
    rs.set_llm('dialogue', {'temperature': 0.9})
    path = tmp_path / 'data' / 'control_center' / 'live_overrides.json'
    assert path.exists()
    rs.save_to_config()
    assert not path.exists(), 'values now live in config.ini'
