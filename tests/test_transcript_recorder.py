import json
import os
from threading import Lock
from types import SimpleNamespace as NS

import pytest

import src.transcripts.recorder as recorder
import src.usage_logger as usage_logger
from src.llm.client_base import ClientBase, VisionMode
from src.llm.claude_cache_connector import ClaudeCacheConnector


@pytest.fixture
def recording(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_logger, '_log_path', str(tmp_path / 'usage.jsonl'))
    monkeypatch.setattr(recorder, '_turn_counters', {})
    recorder.configure(True, str(tmp_path))
    usage_logger.new_conversation()
    usage_logger.set_context(speaker='Lydia', npcs=['Lydia'], has_player=True)
    yield tmp_path
    recorder.configure(False, None)


def _records(folder):
    files = os.listdir(os.path.join(folder, 'data', 'transcripts'))
    assert len(files) == 1
    with open(os.path.join(folder, 'data', 'transcripts', files[0]), encoding='utf-8') as f:
        return [json.loads(line) for line in f]


class LLMClient(ClientBase):
    """Named like the real dialogue client so usage_logger.call_type_for() returns 'dialogue'."""


def _client(chunks):
    client = object.__new__(LLMClient)
    client._generation_lock = Lock()
    client._model_name = 'gpt-test'
    client._base_url = 'https://api.openai.com/v1'
    client._startup_async_client = None
    client._request_params = {'max_tokens': 250, 'stop': ['#']}
    client._image_client = None
    client._function_client = None
    client._enable_vision_next_call = False
    client._vision_mode = VisionMode.DISABLED
    client._is_local = False
    client._caching_enabled = False
    client._claude_cache = ClaudeCacheConnector()

    class FakeStream:
        def __init__(self):
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    class FakeAsyncClient:
        def __init__(self):
            self.kwargs = None
            self.chat = NS(completions=NS(create=self.create))

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return FakeStream()

        async def close(self):
            pass

    fake = FakeAsyncClient()
    client.generate_async_client = lambda: fake
    return client, fake


def _chunk(content=None, tool_calls=None, usage=None):
    choices = [NS(delta=NS(content=content, tool_calls=tool_calls), finish_reason=None)] if (content or tool_calls) else []
    return NS(choices=choices, usage=usage)


class _Thread:
    def __init__(self, messages):
        self._messages = messages

    def get_openai_messages(self):
        return self._messages

    def get_last_message(self):
        return None


def test_disabled_returns_null_turn(tmp_path):
    recorder.configure(False, str(tmp_path))
    turn = recorder.start_turn(LLMClient.__new__(LLMClient), 'm', 'u', [], {}, False)
    assert turn is recorder.NULL_TURN
    turn.finish('x', None, None, False)
    assert not os.path.exists(tmp_path / 'data' / 'transcripts')


def test_non_dialogue_calls_are_not_recorded(recording):
    class SummaryLLMClient:
        pass
    assert recorder.start_turn(SummaryLLMClient(), 'm', 'u', [], {}, False) is recorder.NULL_TURN


def test_strip_images_replaces_base64_only():
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'look'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + 'A' * 5000, 'detail': 'low'}},
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/a.png'}},
    ]}]
    clean, count = recorder.strip_images(messages)
    assert count == 1
    assert 'base64' not in json.dumps(clean).replace('chars base64', '')
    assert clean[0]['content'][1]['omitted'] is True
    assert clean[0]['content'][2]['image_url']['url'] == 'https://example.com/a.png'
    assert messages[0]['content'][1]['image_url']['url'].startswith('data:')  # original untouched


@pytest.mark.asyncio
async def test_streaming_call_records_request_response_timing_usage(recording):
    usage = NS(prompt_tokens=1200, completion_tokens=9, prompt_tokens_details=NS(cached_tokens=1024),
               completion_tokens_details=NS(reasoning_tokens=0))
    tool_delta = [NS(index=0, id='call_1', function=NS(name='Wait', arguments='{"source": ["Lydia"]}'))]
    client, fake = _client([_chunk('As you wish, '), _chunk('my Thane.'), _chunk(tool_calls=tool_delta), _chunk(usage=usage)])
    messages = [{'role': 'system', 'content': 'You are Lydia.'}, {'role': 'user', 'content': 'Wait here.'}]
    tools = [{'type': 'function', 'function': {'name': 'Wait', 'description': 'Make NPC(s) wait.'}}]
    out = [item async for item in client.streaming_call(_Thread(messages), False, tools)]
    assert ('content', 'As you wish, ') in out

    [record] = _records(recording)
    assert record['conversation_id'] == usage_logger._context['conversation_id']
    assert record['call_type'] == 'dialogue' and record['speaker'] == 'Lydia' and record['has_player'] is True
    assert record['request']['messages'] == messages
    assert record['request']['tools'] == tools
    assert record['request']['params'] == {'max_tokens': 250, 'stop': ['#']}  # tools/stream_options not duplicated
    assert record['response']['content'] == 'As you wish, my Thane.'
    assert record['response']['tool_calls'][0]['function']['name'] == 'Wait'
    assert record['response']['completed'] is True and record['response']['error'] is None
    assert record['timing']['ttft_s'] is not None and record['timing']['total_s'] >= record['timing']['ttft_s']
    assert record['usage'] == {'prompt_tokens': 1200, 'cached_tokens': 1024, 'completion_tokens': 9,
                               'reasoning_tokens': 0, 'estimated': False}
    assert record['turn_index'] == 0
    # usage.jsonl still written, same conversation id
    with open(usage_logger._log_path, encoding='utf-8') as f:
        assert json.loads(f.readline())['conversation_id'] == record['conversation_id']


@pytest.mark.asyncio
async def test_followup_call_is_flagged_and_turns_are_numbered(recording):
    usage = NS(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=None, completion_tokens_details=None)
    messages = [{'role': 'system', 'content': 'x'}, {'role': 'user', 'content': 'Wait.'},
                {'role': 'assistant', 'content': None, 'tool_calls': [{'id': '1', 'type': 'function', 'function': {'name': 'Wait', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': '1', 'content': 'done'}]
    for _ in range(2):
        client, _ = _client([_chunk('Aye.'), _chunk(usage=usage)])
        [item async for item in client.streaming_call(_Thread(messages), False, None)]
    records = _records(recording)
    assert [r['turn_index'] for r in records] == [0, 1]
    assert records[0]['flags']['followup_after_tool'] is True


@pytest.mark.asyncio
async def test_api_error_is_recorded(recording, monkeypatch):
    monkeypatch.setattr('src.utils.play_error_sound', lambda: None)
    client, fake = _client([])

    async def boom(**kwargs):
        raise RuntimeError('rate limited')
    fake.chat.completions.create = boom
    client.get_count_tokens = lambda m: 5
    [item async for item in client.streaming_call(_Thread([{'role': 'user', 'content': 'hi'}]), False, None)]
    [record] = _records(recording)
    assert 'rate limited' in record['response']['error']
    assert record['response']['completed'] is False


def test_prompt_vars_are_captured_into_records(tmp_path):
    """play integration: Context passes its prompt variables through capture_prompt_vars."""
    import src.transcripts.recorder as rec
    rec.configure(True, str(tmp_path))
    try:
        assert rec.capture_prompt_vars({'name': 'Lydia', 'current_day': 3}) == {'name': 'Lydia', 'current_day': 3}
        turn = rec.TranscriptTurn('dialogue', 'gpt-4.1-mini', 'https://api.openai.com/v1', [], {}, False)
        assert turn.record['prompt_vars'] == {'name': 'Lydia', 'current_day': 3}
    finally:
        rec.configure(False, None)
        rec._prompt_vars = None
