"""feat/no-hang-listen: interruptible speech waits and a single microphone listener."""
import queue
import sys
import threading
import time
import types

import pytest

from src.stt import listen_control
from src.stt.listen_control import ListenAborted
from src.stt.stt import Transcriber


@pytest.fixture(autouse=True)
def _reset_listen_control():
    listen_control.reset_for_tests()
    yield
    listen_control.reset_for_tests()


# ---------- wait_for_event ----------

def test_wait_returns_true_when_event_set():
    event = threading.Event()
    threading.Timer(0.1, event.set).start()
    assert listen_control.wait_for_event(event, None) is True


def test_wait_returns_false_on_plain_timeout():
    start = time.monotonic()
    assert listen_control.wait_for_event(threading.Event(), 0.3) is False
    assert 0.25 <= time.monotonic() - start < 1.0


def test_max_wait_raises_timeout_when_player_silent():
    start = time.monotonic()
    with pytest.raises(ListenAborted) as aborted:
        listen_control.wait_for_event(threading.Event(), None, max_wait=0.4)
    assert aborted.value.reason == listen_control.ABORT_TIMEOUT
    assert time.monotonic() - start < 1.0


def test_max_wait_does_not_cut_off_a_speaking_player():
    event = threading.Event()
    threading.Timer(0.8, event.set).start()
    # the player is speaking -> the max wait must not fire although 0.2 s passed
    assert listen_control.wait_for_event(event, None, max_wait=0.2, is_speaking=lambda: True) is True


def test_newer_request_supersedes_wait():
    result = {}

    def waiter(request_id):
        with listen_control.request_scope(request_id):
            try:
                listen_control.wait_for_event(threading.Event(), None)
            except ListenAborted as e:
                result['reason'] = e.reason

    first = listen_control.begin_request()
    thread = threading.Thread(target=waiter, args=(first,))
    thread.start()
    time.sleep(0.3)
    assert thread.is_alive()
    listen_control.begin_request()  # the game sent something new
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result['reason'] == listen_control.ABORT_SUPERSEDED


def test_supersede_before_wait_starts_is_not_lost():
    first = listen_control.begin_request()
    listen_control.begin_request()
    with listen_control.request_scope(first):
        with pytest.raises(ListenAborted):
            listen_control.wait_for_event(threading.Event(), None)


def test_shutdown_aborts_wait():
    threading.Timer(0.2, listen_control.shutdown).start()
    with pytest.raises(ListenAborted) as aborted:
        listen_control.wait_for_event(threading.Event(), None)
    assert aborted.value.reason == listen_control.ABORT_SHUTDOWN


# ---------- Transcriber.get_latest_transcription ----------

def _bare_transcriber(max_wait: float = 0) -> Transcriber:
    """A Transcriber without models / microphone: only the state the wait needs."""
    t = object.__new__(Transcriber)
    t.loglevel = 27
    t._transcription_ready = threading.Event()
    t._lock = threading.Lock()
    t._speech_detected = False
    t._current_transcription = ''
    t._temporary_pause_override = None
    t.play_cough_sound = False
    t.player_speech_max_wait = max_wait
    return t


def test_transcriber_returns_speech():
    t = _bare_transcriber(max_wait=5)

    def speak():
        with t._lock:
            t._current_transcription = 'Hello there.'
        t._transcription_ready.set()
    threading.Timer(0.2, speak).start()
    assert t.get_latest_transcription() == 'Hello there.'


def test_transcriber_max_wait():
    t = _bare_transcriber(max_wait=0.4)
    start = time.monotonic()
    with pytest.raises(ListenAborted) as aborted:
        t.get_latest_transcription()
    assert aborted.value.reason == listen_control.ABORT_TIMEOUT
    assert time.monotonic() - start < 1.5


def test_transcriber_silence_auto_response_still_returns_none():
    t = _bare_transcriber(max_wait=5)
    assert t.get_latest_transcription(silence_timeout=0.3) is None


def test_transcriber_zero_max_wait_waits_until_superseded():
    t = _bare_transcriber(max_wait=0)
    first = listen_control.begin_request()
    threading.Timer(0.5, listen_control.begin_request).start()
    with listen_control.request_scope(first):
        with pytest.raises(ListenAborted) as aborted:
            t.get_latest_transcription()
    assert aborted.value.reason == listen_control.ABORT_SUPERSEDED


# ---------- single microphone listener ----------

class _FakeInputStream:
    opened = 0

    def __init__(self, **kwargs):
        _FakeInputStream.opened += 1
        self.closed = False

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        self.closed = True


def _listening_transcriber() -> Transcriber:
    t = object.__new__(Transcriber)
    t.loglevel = 27
    t._running = False
    t._stream = None
    t._processing_thread = None
    t._audio_queue = queue.Queue()
    t._lock = threading.Lock()
    t._speech_detected = False
    t.ptt_enabled = False
    t._reset_state = lambda: None
    return t


def test_only_one_listener_runs(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(InputStream=_FakeInputStream))
    old, new = _listening_transcriber(), _listening_transcriber()
    old.start_listening('prompt')
    assert old.is_listening
    old_stream = old._stream

    new.start_listening('prompt')
    assert new.is_listening
    assert not old.is_listening  # the stale listener was stopped
    assert old_stream.closed

    new.start_listening('prompt')  # idempotent: no second stream for the same Transcriber
    assert _FakeInputStream.opened == 2
    new.stop_listening()
    assert not new.is_listening
