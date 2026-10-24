"""API tests for the Control Center (/control, /api/control/*, /api/usage) via FastAPI TestClient."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.runtime_settings as runtime_settings
import src.usage_logger as usage_logger
from src.config.config_loader import ConfigLoader
from src.config.definitions.game_definitions import GameEnum
from src.control_center import usage
from src.control_center.route import ControlCenterRoute


@pytest.fixture
def usage_file(tmp_path, monkeypatch):
    path = tmp_path / 'usage.jsonl'
    rows = [
        {'ts': '2026-10-24T18:00:00', 'conversation_id': 'c1', 'call_type': 'dialogue', 'model': 'gpt-4.1-mini',
         'speaker': 'Lydia', 'npcs': ['Lydia'], 'prompt_tokens': 3000, 'cached_tokens': 2000, 'completion_tokens': 40, 'latency_s': 1.4},
        {'ts': '2026-10-24T18:00:30', 'conversation_id': 'c1', 'call_type': 'memory', 'model': 'gpt-4.1-mini',
         'speaker': 'Lydia', 'prompt_tokens': 1500, 'cached_tokens': 0, 'completion_tokens': 200, 'latency_s': 3.0},
        {'ts': '2026-10-24T18:01:00', 'conversation_id': 'c2', 'call_type': 'dialogue', 'model': 'mystery-model',
         'speaker': 'Nazeem', 'prompt_tokens': 100, 'cached_tokens': 0, 'completion_tokens': 10, 'latency_s': 0.9},
    ]
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\nnot json\n', encoding='utf-8')
    monkeypatch.setattr(usage_logger, '_log_path', str(path))
    return path


@pytest.fixture
def client(tmp_path, monkeypatch, usage_file):
    monkeypatch.setattr(runtime_settings, '_settings', runtime_settings.RuntimeSettings())
    from src.actions.function_manager import FunctionManager
    monkeypatch.setattr(FunctionManager, 'load_all_actions', staticmethod(lambda disabled_actions=None: None))
    config = ConfigLoader(mygame_folder_path=str(tmp_path / 'cfg'), game_override=GameEnum.SKYRIM) if (tmp_path / 'cfg').mkdir() is None else None
    app = FastAPI()
    route = ControlCenterRoute(config)
    route.add_route_to_server(app)
    test_client = TestClient(app)
    test_client.config = config
    return test_client


def test_page_served(client):
    res = client.get('/control')
    assert res.status_code == 200
    assert 'Mantella Control Center' in res.text
    assert 'http://' not in res.text.split('<script>')[0].replace('http://localhost', ''), 'no external resources'
    assert 'cdn' not in res.text.lower()


def test_state_shows_config_values(client):
    state = client.get('/api/control/state').json()
    assert state['config']['dialogue']['max_tokens'] == 250
    assert state['config']['general']['max_response_sentences_single'] == 4
    assert state['live'] == {'llm': {}, 'general': {}, 'npc': {}}
    assert 'Brawl' in state['actions']
    assert 'gpt-4.1-mini' in state['models']
    assert state['npc_suggestions'] == ['Lydia', 'Nazeem']
    assert state['last_call']['speaker'] == 'Nazeem'


def test_live_reset_and_effective(client):
    res = client.post('/api/control/live', json={'scope': 'dialogue', 'values': {'temperature': 1.1, 'model': 'gpt-5-mini'}})
    assert res.status_code == 200
    state = res.json()
    assert state['live']['llm']['dialogue'] == {'temperature': 1.1, 'model': 'gpt-5-mini'}
    assert state['effective']['dialogue']['model'] == 'gpt-5-mini'
    assert state['effective']['dialogue']['max_tokens'] == 250
    assert runtime_settings.llm_overrides('dialogue')['temperature'] == 1.1

    state = client.post('/api/control/live', json={'scope': 'npc', 'npc': 'Lydia', 'values': {'temperature': 0.3}}).json()
    assert state['live']['npc'] == {'Lydia': {'temperature': 0.3}}
    assert runtime_settings.llm_overrides('dialogue', 'Lydia')['temperature'] == 0.3

    state = client.post('/api/control/live', json={'scope': 'general', 'values': {'max_response_sentences_single': 2, 'disabled_actions': ['Brawl']}}).json()
    assert state['effective']['general']['max_response_sentences_single'] == 2
    assert state['effective']['general']['disabled_actions'] == ['Brawl']

    state = client.post('/api/control/reset', json={'scope': 'dialogue'}).json()
    assert state['live']['llm'] == {} and state['live']['npc'] and state['live']['general']
    state = client.post('/api/control/reset', json={'scope': 'all'}).json()
    assert state['live'] == {'llm': {}, 'general': {}, 'npc': {}}


@pytest.mark.parametrize('body', [
    {'scope': 'dialogue', 'values': {'temperature': 9}},
    {'scope': 'dialogue', 'values': {'api_key': 'x'}},
    {'scope': 'nope', 'values': {}},
    {'scope': 'npc', 'npc': '', 'values': {'temperature': 1}},
    {'scope': 'general', 'values': {'vision_enabled': 'yes'}},
])
def test_live_rejects_bad_input(client, body):
    res = client.post('/api/control/live', json=body)
    assert res.status_code == 400
    assert 'error' in res.json()
    assert runtime_settings.get().snapshot()['live'] == {'llm': {}, 'general': {}, 'npc': {}}


def test_save_to_config(client):
    client.post('/api/control/live', json={'scope': 'memory', 'values': {'max_tokens': 800}})
    client.post('/api/control/live', json={'scope': 'general', 'values': {'max_response_sentences_multi': 5}})
    res = client.post('/api/control/save')
    assert res.status_code == 200
    assert set(res.json()['saved']) == {'summary_llm_params', 'max_response_sentences_multi'}
    text = (client.config.save_folder and open(client.config._ConfigLoader__file_name, encoding='utf-8').read())
    assert 'max_response_sentences_multi = 5' in text
    assert res.json()['state']['live']['llm'] == {}


def test_usage_api_prices_records(client, usage_file):
    data = client.get('/api/usage').json()
    assert data['log_path'] == str(usage_file)
    assert len(data['records']) == 3
    first = data['records'][0]
    # gpt-4.1-mini: 1000 uncached * 0.40 + 2000 cached * 0.10 + 40 out * 1.60 per 1M
    assert first['cost_usd'] == pytest.approx((1000 * 0.40 + 2000 * 0.10 + 40 * 1.60) / 1e6)
    assert data['records'][2]['cost_usd'] is None
    assert data['eur_per_usd'] > 0


def test_usage_cache_rereads_on_change(usage_file):
    first = usage.load_records(str(usage_file))
    assert usage.load_records(str(usage_file)) is first
    with open(usage_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'ts': '2026-10-24T18:02:00', 'call_type': 'dialogue', 'model': 'gpt-4.1', 'prompt_tokens': 1, 'completion_tokens': 1}) + '\n')
    assert len(usage.load_records(str(usage_file))) == 4


def test_missing_usage_file(client, monkeypatch, tmp_path):
    monkeypatch.setattr(usage_logger, '_log_path', str(tmp_path / 'nothing.jsonl'))
    assert client.get('/api/usage').json()['records'] == []
    assert client.get('/api/control/state').json()['last_call'] is None


# ---------- Memories ----------

def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


@pytest.fixture
def conv_root(client):
    from src.control_center import memories
    root = memories.conversations_root(client.config)
    from pathlib import Path
    root = Path(root)
    _write(root / 'Vaska1' / 'Lydia - 0A2C94' / 'Lydia_summary_1.txt',
           '[Day 1, 2 in the afternoon]\nLydia swore to carry the burdens of Vaska.\n\n[Day 2, 8 in the evening]\nVaska insulted Lydia in front of Alfhild.\n')
    _write(root / 'Vaska1' / 'Nazeem - 013BBF' / 'Nazeem_summary_1.txt', 'Nazeem asked whether Vaska gets to the Cloud District very often.\n')
    _write(root / 'Vaska1' / 'Whiterun Guard' / 'Whiterun Guard_summary_1.txt', 'Old legacy folder memory.\n')
    _write(root / 'Vaska1' / 'Empty - 000001' / 'Empty.json', '[]')
    return root


def test_memories_list_and_search(client, conv_root):
    npcs = client.get('/api/memories/npcs').json()['npcs']
    assert {n['name'] for n in npcs} == {'Lydia', 'Nazeem', 'Whiterun Guard'}, 'folders without summaries are skipped'
    legacy = next(n for n in npcs if n['name'] == 'Whiterun Guard')
    assert legacy['ref_id'] is None and legacy['id'] == 'Vaska1/Whiterun Guard'
    assert [n['name'] for n in client.get('/api/memories/npcs', params={'q': 'lyd'}).json()['npcs']] == ['Lydia']
    assert client.get('/api/memories/npcs', params={'q': '013bbf'}).json()['npcs'][0]['name'] == 'Nazeem'


def test_memories_npc_entries_newest_first_with_day(client, conv_root):
    data = client.get('/api/memories/npc', params={'id': 'Vaska1/Lydia - 0A2C94'}).json()
    assert data['name'] == 'Lydia' and data['ref_id'] == '0A2C94'
    assert [e['day'] for e in data['entries']] == ['Day 2, 8 in the evening', 'Day 1, 2 in the afternoon']
    assert data['entries'][0]['text'] == 'Vaska insulted Lydia in front of Alfhild.'
    assert data['relationship'] is None


def test_memories_condensed_files_and_new_entries(client, conv_root):
    lydia = conv_root / 'Vaska1' / 'Lydia - 0A2C94'
    _write(lydia / 'Lydia_summary_2.txt', 'Condensed: Lydia serves Vaska but resents his rudeness.\n')
    data = client.get('/api/memories/npc', params={'id': 'Vaska1/Lydia - 0A2C94'}).json()
    assert data['entries'][0]['text'].startswith('Condensed') and not data['entries'][0]['condensed']
    assert data['entries'][1]['condensed']
    ids = {e['id'] for e in data['entries']}
    # a summary appended at conversation end shows up on the next poll (mtime/size change)
    with open(lydia / 'Lydia_summary_2.txt', 'a', encoding='utf-8') as f:
        f.write('[Day 3, 9 in the morning]\nVaska apologised to Lydia.\n')
    data = client.get('/api/memories/npc', params={'id': 'Vaska1/Lydia - 0A2C94'}).json()
    new = [e for e in data['entries'] if e['id'] not in ids]
    assert len(new) == 1 and new[0]['day'] == 'Day 3, 9 in the morning' and data['entries'][0] == new[0]


def test_memories_relationship_slot(client, conv_root):
    _write(conv_root / 'Vaska1' / 'Lydia - 0A2C94' / 'relationship.json',
           json.dumps({'opinion_of_player': -35, 'grudges': ['called her names in front of Alfhild']}))
    data = client.get('/api/memories/npc', params={'id': 'Vaska1/Lydia - 0A2C94'}).json()
    assert data['relationship'] == {'opinion_of_player': -35, 'grudges': ['called her names in front of Alfhild']}
    _write(conv_root / 'Vaska1' / 'Nazeem - 013BBF' / 'Nazeem_relationship.json', '{broken')
    assert '_error' in client.get('/api/memories/npc', params={'id': 'Vaska1/Nazeem - 013BBF'}).json()['relationship']


@pytest.mark.parametrize('bad', ['../etc', 'Vaska1/../../x', 'Vaska1', '', 'Vaska1/Nobody', '../../cfg'])
def test_memories_npc_rejects_bad_ids(client, conv_root, bad):
    assert client.get('/api/memories/npc', params={'id': bad}).status_code == 404


def test_memories_active_from_conversation_context(client, conv_root, monkeypatch):
    monkeypatch.setattr(usage_logger, '_context', {'conversation_id': 'c9', 'npcs': ['Lydia', 'Whiterun Guard 2', 'Hadvar']})
    data = client.get('/api/memories/active').json()
    assert data['source'] == 'conversation' and data['conversation_id'] == 'c9'
    assert [n['name'] for n in data['npcs']] == ['Lydia', 'Whiterun Guard']
    assert data['no_memories_yet'] == ['Hadvar']


def test_memories_active_falls_back_to_last_logged_conversation_then_recent(client, conv_root, monkeypatch):
    monkeypatch.setattr(usage_logger, '_context', {})
    data = client.get('/api/memories/active').json()  # usage fixture: last dialogue with npcs is Lydia
    assert data['source'] == 'last' and [n['name'] for n in data['npcs']] == ['Lydia']
    monkeypatch.setattr(usage_logger, '_log_path', str(conv_root / 'none.jsonl'))
    data = client.get('/api/memories/active').json()
    assert data['source'] == 'recent' and len(data['npcs']) == 3
