"""Control Center: /control page plus a small JSON API on Mantella's own FastAPI app.

  GET  /control                 the page (static HTML + JS, no CDN)
  GET  /api/control/state       config / live / effective values, actions, model + NPC suggestions
  POST /api/control/live        {"scope": "dialogue"|"memory"|"general"|"npc", "npc": "...", "values": {...}}
  POST /api/control/reset       {"scope": "all"|"dialogue"|"memory"|"general"|"npc", "npc": "..."}
  POST /api/control/save        write live dialogue/memory/general values to config.ini
  GET  /api/usage               usage.jsonl records with cost_usd (Spend tab)
  GET  /api/memories/active     memories of the NPCs in the current (or last) conversation
  GET  /api/memories/npcs?q=    all NPCs with memories (search), newest first
  GET  /api/memories/npc?id=    one NPC's memories ('<world>/<folder>'), read-only
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import src.runtime_settings as runtime_settings
from src.config.config_loader import ConfigLoader
from src.control_center import memories, usage
from src.http.routes.routeable import routeable
import src.utils as utils

logger = utils.get_logger()

PAGE_PATH = Path(__file__).resolve().parent / 'static' / 'control.html'
NO_STORE = {'Cache-Control': 'no-store'}


def _all_action_names() -> list[str]:
    try:
        from src.config.definitions.action_definitions import _scan_action_names
        return sorted(_scan_action_names())
    except Exception:
        return []


def _last_call(records: list[dict]) -> dict | None:
    for r in reversed(records):
        if r.get('call_type') == 'dialogue':
            return r
    return records[-1] if records else None


class ControlCenterRoute(routeable):
    def __init__(self, config: ConfigLoader) -> None:
        super().__init__(config)
        self._settings = runtime_settings.get()
        self._settings.attach_config(config)

    def _state(self) -> dict[str, Any]:
        state = self._settings.snapshot()
        records = usage.load_records()
        pricing_models = usage.known_models()
        configured = [m for m in (state['config'].get('dialogue', {}).get('model'), state['config'].get('memory', {}).get('model')) if m]
        speakers = sorted({r.get('speaker') for r in records[-2000:] if isinstance(r.get('speaker'), str) and r.get('call_type') == 'dialogue'})
        state.update({
            'actions': _all_action_names(),
            'models': sorted(set(pricing_models + configured)),
            'reasoning_efforts': list(runtime_settings.REASONING_EFFORTS),
            'npc_suggestions': speakers,
            'last_call': _last_call(records),
            'eur_per_usd': usage.eur_per_usd(),
        })
        return state

    def add_route_to_server(self, app: FastAPI):
        @app.get('/control', response_class=HTMLResponse)
        async def control_page():
            return HTMLResponse(PAGE_PATH.read_text(encoding='utf-8'), headers=NO_STORE)

        @app.get('/api/control/state')
        async def control_state():
            return JSONResponse(self._state(), headers=NO_STORE)

        @app.post('/api/control/live')
        async def control_live(request: Request):
            body = await request.json()
            scope = body.get('scope')
            values = body.get('values') or {}
            if not isinstance(values, dict):
                return JSONResponse({'error': 'values must be an object'}, status_code=400)
            try:
                if scope in runtime_settings.LLM_CALL_TYPES:
                    self._settings.set_llm(scope, values)
                elif scope == 'general':
                    self._settings.set_general(values)
                elif scope == 'npc':
                    self._settings.set_npc(body.get('npc') or '', values)
                else:
                    return JSONResponse({'error': f'unknown scope {scope!r}'}, status_code=400)
            except runtime_settings.ValidationError as e:
                return JSONResponse({'error': str(e)}, status_code=400)
            return JSONResponse(self._state(), headers=NO_STORE)

        @app.post('/api/control/reset')
        async def control_reset(request: Request):
            body = await request.json()
            try:
                self._settings.reset(body.get('scope', 'all'), body.get('npc'))
            except runtime_settings.ValidationError as e:
                return JSONResponse({'error': str(e)}, status_code=400)
            return JSONResponse(self._state(), headers=NO_STORE)

        @app.post('/api/control/save')
        async def control_save():
            try:
                written = self._settings.save_to_config()
            except (runtime_settings.ValidationError, RuntimeError) as e:
                return JSONResponse({'error': str(e)}, status_code=400)
            return JSONResponse({'saved': written, 'state': self._state()}, headers=NO_STORE)

        @app.get('/api/usage')
        async def usage_data():
            return JSONResponse(usage.usage_payload(), headers=NO_STORE)

        @app.get('/api/memories/active')
        async def memories_active():
            return JSONResponse(memories.active_payload(self._memories_root()), headers=NO_STORE)

        @app.get('/api/memories/npcs')
        async def memories_npcs(q: str = ''):
            return JSONResponse({'npcs': memories.list_npcs(self._memories_root(), q)}, headers=NO_STORE)

        @app.get('/api/memories/npc')
        async def memories_npc(id: str = ''):
            data = memories.npc_memories(self._memories_root(), id)
            if data is None:
                return JSONResponse({'error': 'unknown NPC'}, status_code=404)
            return JSONResponse(data, headers=NO_STORE)

        port = getattr(self._config, 'port', 4999)
        logger.log(24, '\nMantella Control Center (live tuning + spend):')
        logger.log(25, f'http://localhost:{port}/control')

    def _memories_root(self) -> str:
        return memories.conversations_root(self._config)

    def _setup_route(self):
        pass
