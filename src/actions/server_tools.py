"""Registry of server-side tools: actions the LLM can call that are executed by the Mantella server
itself and never forwarded to the game (eg RememberAboutPlayer).

A feature registers a handler (and optionally an availability check) for an action identifier:

    server_tools.register("mantella_remember_about_player", handler, is_available)

- ``handler(arguments: dict, characters) -> str`` runs the tool and returns a short result text.
- ``is_available(context) -> bool`` decides per conversation whether the tool is offered to the LLM.

FunctionManager calls ``is_available`` while building the tool list and ``run`` while parsing tool
calls (server tools are removed from what is sent to the game). Results are kept per tool_call_id
so that ``output_manager`` can later return them as real ToolMessage contents instead of "done"
(``pop_result``); until then the conversation simply continues with the generic "done" result.
"""
from __future__ import annotations

from typing import Any, Callable

from src import utils

logger = utils.get_logger()

Handler = Callable[[dict, Any], str]
Availability = Callable[[Any], bool]

_handlers: dict[str, Handler] = {}
_availability: dict[str, Availability] = {}
_results: dict[str, str] = {}
MAX_STORED_RESULTS = 50


def register(identifier: str, handler: Handler, is_available: Availability | None = None) -> None:
    _handlers[identifier] = handler
    if is_available:
        _availability[identifier] = is_available


def is_server_tool(identifier: str) -> bool:
    return identifier in _handlers


def is_available(identifier: str, context) -> bool:
    """True if the tool should be offered in this conversation (non-server tools: always True)."""
    check = _availability.get(identifier)
    if check is None:
        return True
    try:
        return bool(check(context))
    except Exception as e:
        logger.error(f"Availability check for server tool '{identifier}' failed: {e}")
        return False


def run(identifier: str, arguments: dict | None, characters=None, tool_call_id: str = "") -> str:
    """Execute a server tool. Never raises; returns the result text."""
    try:
        result = _handlers[identifier](arguments or {}, characters)
    except Exception as e:
        logger.error(f"Server tool '{identifier}' failed: {e}")
        result = f"error: {e}"
    if tool_call_id:
        if len(_results) >= MAX_STORED_RESULTS:
            _results.pop(next(iter(_results)))
        _results[tool_call_id] = result
    return result


def pop_result(tool_call_id: str, default: str = "done") -> str:
    return _results.pop(tool_call_id, default)
