"""Keeps the wait for player speech from hanging the server (feat/no-hang-listen).

Upstream waited for the player's voice with ``threading.Event.wait()`` inside the async
``/mantella`` route, i.e. on the event loop thread. While the player stayed silent nothing
else could be served (/ui, /control, /api/*), and when the game abandoned the conversation
(stopquest, recast, crash, loading a save) the wait never ended and the process ignored
SIGTERM.

This module holds the process-wide state that lets a wait end early:

* every ``/mantella`` request gets a sequence number (``begin_request``); a wait that belongs
  to an older request than the newest one is *superseded* (the game only ever has one request
  in flight, so a new request means the old one was abandoned) and raises ``ListenAborted``;
* ``shutdown()`` (called from the SIGTERM/SIGINT handler) aborts every wait;
* ``wait_for_event`` replaces ``Event.wait`` with short polling slices plus an optional
  maximum wait for the player to start speaking (config ``player_speech_max_wait``);
* ``claim_listener`` makes sure only one Transcriber has an open microphone stream.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

POLL_SECONDS = 0.2

ABORT_SUPERSEDED = 'superseded'
ABORT_TIMEOUT = 'timeout'
ABORT_SHUTDOWN = 'shutdown'


class ListenAborted(Exception):
    """Raised inside a speech wait that must end without a transcription."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_state_lock = threading.Lock()
_latest_request: int = 0
_shutting_down = threading.Event()
_local = threading.local()
_active_listener: Any = None


def begin_request() -> int:
    """Register a new game request. Any wait owned by an older request is aborted."""
    global _latest_request
    with _state_lock:
        _latest_request += 1
        return _latest_request


def latest_request() -> int:
    return _latest_request


@contextmanager
def request_scope(request_id: int) -> Iterator[None]:
    """Bind the current (worker) thread to a request so waits in it can be superseded."""
    previous = getattr(_local, 'request_id', None)
    _local.request_id = request_id
    try:
        yield
    finally:
        _local.request_id = previous


def abort_reason() -> str | None:
    """Why a wait in the current thread should stop now, or None to keep waiting."""
    if _shutting_down.is_set():
        return ABORT_SHUTDOWN
    request_id = getattr(_local, 'request_id', None)
    if request_id is not None and request_id < _latest_request:
        return ABORT_SUPERSEDED
    return None


def check_abort() -> None:
    reason = abort_reason()
    if reason:
        raise ListenAborted(reason)


def shutdown() -> None:
    """Abort every wait (server is shutting down). Safe to call from a signal handler."""
    _shutting_down.set()


def is_shutting_down() -> bool:
    return _shutting_down.is_set()


def wait_for_event(event: threading.Event, timeout: float | None, *,
                   max_wait: float = 0, started_at: float | None = None,
                   is_speaking: Callable[[], bool] = lambda: False) -> bool:
    """``event.wait(timeout)`` that can be interrupted.

    Returns True when the event is set, False when ``timeout`` (if not None) elapsed.
    Raises ``ListenAborted`` when the request was superseded, the server is shutting down,
    or ``max_wait`` (> 0) seconds passed since ``started_at`` without the player speaking.
    """
    now = time.monotonic()
    started_at = now if started_at is None else started_at
    timeout_end = None if timeout is None else now + timeout
    while True:
        check_abort()
        now = time.monotonic()
        if max_wait > 0 and now - started_at >= max_wait and not is_speaking():
            raise ListenAborted(ABORT_TIMEOUT)
        slice_seconds = POLL_SECONDS
        if timeout_end is not None:
            remaining = timeout_end - now
            if remaining <= 0:
                return event.is_set()
            slice_seconds = min(slice_seconds, remaining)
        if event.wait(slice_seconds):
            return True


def claim_listener(listener: Any) -> Any:
    """Make ``listener`` the only active microphone listener; returns the previous one."""
    global _active_listener
    with _state_lock:
        previous = _active_listener
        _active_listener = listener
    return previous if previous is not listener else None


def release_listener(listener: Any) -> None:
    global _active_listener
    with _state_lock:
        if _active_listener is listener:
            _active_listener = None


def reset_for_tests() -> None:
    global _latest_request, _active_listener
    with _state_lock:
        _latest_request = 0
        _active_listener = None
    _shutting_down.clear()
    _local.request_id = None
