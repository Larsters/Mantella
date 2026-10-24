"""feat/no-hang-listen: /mantella waiting for player speech must not block the server.

Uses the real mantella_route and GameStateManager.player_input with a fake conversation whose
`process_player_input` waits exactly like Transcriber.get_latest_transcription does
(listen_control.wait_for_event on an event nobody sets = a player who never speaks).
No models, microphone, LLM or TTS are involved.
"""
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.game_manager import GameStateManager
from src.http import http_server as http_server_module
from src.http.communication_constants import communication_constants as comm_consts
from src.http.routes.mantella_route import mantella_route
from src.stt import listen_control

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _reset_listen_control():
    listen_control.reset_for_tests()
    yield
    listen_control.reset_for_tests()


class FakeStt:
    def __init__(self):
        self.stop_calls = 0

    def stop_listening(self):
        self.stop_calls += 1


class SilentPlayerTalk:
    """Conversation stand-in: waits for speech that never comes."""

    def __init__(self, max_wait: float):
        self.max_wait = max_wait
        self.waits_started = threading.Event()

    def process_player_input(self, player_text):
        self.waits_started.set()
        listen_control.wait_for_event(threading.Event(), None, max_wait=self.max_wait)
        raise AssertionError('unreachable: nobody sets the event')


def make_game(max_wait: float = 0):
    config = SimpleNamespace(player_speech_max_wait=max_wait)
    game = object.__new__(GameStateManager)
    talk = SilentPlayerTalk(max_wait)
    stt = FakeStt()
    game._GameStateManager__talk = talk
    game._GameStateManager__stt = stt
    game._GameStateManager__config = config
    game._GameStateManager__first_line = True
    game._GameStateManager__update_context = lambda json: None
    started = []
    game.start_conversation = lambda json: started.append(json) or {
        comm_consts.KEY_REPLYTYPE: comm_consts.KEY_REPLYTTYPE_STARTCONVERSATIONCOMPLETED}
    return game, talk, stt, started


def make_client(game) -> TestClient:
    route = mantella_route(SimpleNamespace(show_http_debug_messages=False), {'language': 'en'})
    route._can_route_be_used = lambda: True
    route._mantella_route__game = game
    app = FastAPI()
    route.add_route_to_server(app)

    @app.get('/ping')
    async def ping():
        return {'ok': True}

    return TestClient(app)


PLAYER_INPUT = {comm_consts.KEY_REQUESTTYPE: comm_consts.KEY_REQUESTTYPE_PLAYERINPUT,
                comm_consts.KEY_REQUESTTYPE_PLAYERINPUT: ''}
START = {comm_consts.KEY_REQUESTTYPE: comm_consts.KEY_REQUESTTYPE_STARTCONVERSATION}


def post_in_thread(client, payload, results, key):
    def run():
        results[key] = client.post('/mantella', json=payload).json()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_server_responsive_while_waiting_for_speech_and_new_conversation_cancels_wait():
    game, talk, stt, started = make_game(max_wait=0)  # 0 = would wait forever
    results = {}
    with make_client(game) as client:
        waiting = post_in_thread(client, PLAYER_INPUT, results, 'listen')
        assert talk.waits_started.wait(2)

        # other routes answer while /mantella waits for the player's voice
        t0 = time.monotonic()
        for _ in range(5):
            assert client.get('/ping').json() == {'ok': True}
        assert time.monotonic() - t0 < 1.0
        assert waiting.is_alive()

        # the game abandons the conversation and starts a new one
        reply = client.post('/mantella', json=START).json()
        assert reply[comm_consts.KEY_REPLYTYPE] == comm_consts.KEY_REPLYTTYPE_STARTCONVERSATIONCOMPLETED
        waiting.join(2)
        assert not waiting.is_alive()

    assert results['listen'] == {comm_consts.KEY_REPLYTYPE: comm_consts.KEY_REPLYTYPE_PLAYERTALK}
    assert stt.stop_calls == 1  # the stale listener was stopped
    assert len(started) == 1


def test_max_wait_returns_player_talk():
    game, talk, stt, _ = make_game(max_wait=0.5)
    with make_client(game) as client:
        t0 = time.monotonic()
        reply = client.post('/mantella', json=PLAYER_INPUT).json()
        elapsed = time.monotonic() - t0
    # mantella_player_talk = the game's "your turn" reply: it simply asks to listen again
    assert reply == {comm_consts.KEY_REPLYTYPE: comm_consts.KEY_REPLYTYPE_PLAYERTALK}
    assert 0.4 <= elapsed < 2.0
    assert stt.stop_calls == 1


def test_signal_handler_aborts_wait(monkeypatch):
    monkeypatch.setattr(http_server_module, '_force_exit', lambda: None)
    server = http_server_module._MantellaUvicornServer(uvicorn.Config(FastAPI()))
    server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit
    assert listen_control.is_shutting_down()
    with pytest.raises(listen_control.ListenAborted):
        listen_control.wait_for_event(threading.Event(), None)


SERVER_SCRIPT = textwrap.dedent('''
    import sys, threading
    from types import SimpleNamespace
    sys.path.insert(0, sys.argv[2])
    from src.http.http_server import http_server
    from src.http.routes.mantella_route import mantella_route
    from src.stt import listen_control

    class Game:
        def player_input(self, json):
            print('WAITING', flush=True)
            try:
                listen_control.wait_for_event(threading.Event(), None)
            except listen_control.ListenAborted as e:
                print('ABORTED', e.reason, flush=True)
                return {'mantella_reply_type': 'mantella_player_talk'}

    route = mantella_route(SimpleNamespace(show_http_debug_messages=False), {'language': 'en'})
    route._can_route_be_used = lambda: True
    route._mantella_route__game = Game()
    http_server().start(int(sys.argv[1]), [route], play_startup_sound=False)
''')


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.mark.parametrize('sig', [signal.SIGTERM, signal.SIGINT])
def test_real_server_exits_on_signal_mid_wait(tmp_path, sig):
    """Starts a real uvicorn server (temp port, muted, temp HOME), blocks /mantella on a
    speech wait, sends the signal and checks the process exits promptly."""
    import requests
    port = _free_port()
    script = tmp_path / 'server.py'
    script.write_text(SERVER_SCRIPT)
    env = dict(os.environ, MANTELLA_MUTE='1', HOME=str(tmp_path))
    proc = subprocess.Popen([sys.executable, str(script), str(port), str(REPO)], cwd=tmp_path, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                requests.get(f'http://127.0.0.1:{port}/docs', timeout=0.5)
                break
            except requests.RequestException:
                time.sleep(0.2)
        result = {}

        def listen():
            try:
                result['reply'] = requests.post(f'http://127.0.0.1:{port}/mantella', json=PLAYER_INPUT, timeout=20).json()
            except requests.RequestException as e:
                result['error'] = e
        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
        time.sleep(1.0)
        # responsive while the wait is pending
        assert requests.get(f'http://127.0.0.1:{port}/docs', timeout=2).status_code == 200

        t0 = time.monotonic()
        proc.send_signal(sig)
        proc.wait(timeout=8)
        assert time.monotonic() - t0 < 8
        output = proc.stdout.read()
        assert 'WAITING' in output and 'ABORTED shutdown' in output, output
    finally:
        if proc.poll() is None:
            proc.kill()
