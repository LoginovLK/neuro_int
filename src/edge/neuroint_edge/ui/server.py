"""Local web UI: live camera with bones overlay + gesture mapping editor.

Endpoints (loopback only):
  GET /        - index.html (assets/ dir)
  GET /static/*- CSS/JS
  GET /video   - MJPEG stream of the captured frames
  WS   /ws     - UI protocol: state snapshots, landmarks relay, editor ops

This is a host-local control surface; it must never be exposed beyond
127.0.0.1 (see docs/SECURITY.md).
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import typing

from aiohttp import web, web_ws
from aiohttp.client_exceptions import ClientConnectionResetError

from neuroint_edge.gesture_store import GestureStore
from neuroint_edge.key_recorder import ComboRecorder
from neuroint_edge.ui.hub import UiHub

logger = logging.getLogger(__name__)

ASSETS_DIR = pathlib.Path(__file__).resolve().parent / "assets"

_POSES = ("index_up", "two_up", "open_palm", "fist", "pinch")
_DIRECTIONS = ("left", "right", "up", "down")


class UiServer:
    def __init__(
        self,
        hub: UiHub,
        store: GestureStore,
        recorder: ComboRecorder,
        reload_config,  # async () -> None: ask the brain to re-read gestures.yaml
        pause_toggle,  # async () -> None: ask the brain to toggle pause
        set_cursor_prefs,  # (sensitivity, mode) -> dict: apply + persist
        cursor_prefs,  # () -> dict: current cursor settings
        host: str = "127.0.0.1",
        port: int = 8756,
    ) -> None:
        self._hub = hub
        self._store = store
        self._recorder = recorder
        self._reload_config = reload_config
        self._pause_toggle = pause_toggle
        self._set_cursor_prefs = set_cursor_prefs
        self._get_cursor_prefs = cursor_prefs
        self._host = host
        self._port = port

    async def run(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/static/{name}", self._static)
        app.router.add_get("/video", self._mjpeg)
        app.router.add_get("/ws", self._ws)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info("UI ready at http://%s:%d", self._host, self._port)
        await asyncio.Future()

    # --- HTTP ---------------------------------------------------------------

    async def _index(self, request: web.Request) -> web.Response:
        return web.FileResponse(ASSETS_DIR / "index.html")

    async def _static(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if name not in ("style.css", "app.js"):
            raise web.HTTPNotFound
        return web.FileResponse(ASSETS_DIR / name)

    async def _mjpeg(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame"}
        )
        await response.prepare(request)
        while True:
            try:
                frame = self._hub.latest_frame()
                if frame is not None:
                    await response.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                await asyncio.sleep(1.0 / 30.0)
            except (ConnectionResetError, ClientConnectionResetError):
                return response  # browser tab closed/refreshed: normal case

    # --- WebSocket ----------------------------------------------------------

    async def _ws(self, request: web.Request) -> web_ws.WebSocketResponse:
        ws = web_ws.WebSocketResponse()
        await ws.prepare(request)
        self._hub.attach(ws)
        try:
            await self._send_state(ws)
            async for msg in ws:
                if msg.type != typing.cast(object, web.WSMsgType).TEXT:
                    continue
                await self._handle_ui_message(ws, msg.data)
        finally:
            self._hub.detach(ws)
        return ws

    async def _send_state(self, ws) -> None:
        raw = await self._store.read_all()
        state = {
            "type": "state",
            "poses": list(_POSES),
            "directions": list(_DIRECTIONS),
            "actions": self._hub.actions,
            "rules": raw["rules"],
            "custom_actions": raw["custom_actions"],
            "recording": self._recorder.active,
            "cursor_prefs": self._get_cursor_prefs(),
        }
        if self._hub.last_status is not None:
            state["status"] = self._hub.last_status
        await ws.send_json(state)

    async def _handle_ui_message(self, ws, raw: str) -> None:
        import json

        try:
            data = json.loads(raw)
        except ValueError:
            return
        msg_type = data.get("type")
        try:
            if msg_type == "add_rule":
                await self._store.add_rule(
                    data["pose"], data.get("direction", ""), bool(data.get("hold")), data["action"]
                )
                await self._after_change(ws)
            elif msg_type == "remove_rule":
                await self._store.remove_rule(data["pose"], data["motion"], data["action"])
                await self._after_change(ws)
            elif msg_type == "add_custom":
                action_id = await self._store.add_custom_action(
                    data["keys"], data.get("description", "")
                )
                await self._after_change(ws)
                await ws.send_json({"type": "custom_added", "action_id": action_id})
            elif msg_type == "delete_custom":
                await self._store.delete_custom_action(data["action_id"])
                await self._after_change(ws)
            elif msg_type == "record_start":
                self._start_recording(ws)
            elif msg_type == "record_cancel":
                self._recorder.cancel()
                await ws.send_json({"type": "recording", "active": False})
            elif msg_type == "pause_toggle":
                await self._pause_toggle()
            elif msg_type == "set_cursor":
                prefs = self._set_cursor_prefs(
                    float(data["sensitivity"]), str(data.get("mode", "relative"))
                )
                await ws.send_json({"type": "cursor_prefs", **prefs})
            else:
                await ws.send_json(
                    {"type": "notify", "level": "error", "text": f"unknown op {msg_type!r}"}
                )
        except (ValueError, KeyError) as exc:
            await ws.send_json({"type": "notify", "level": "error", "text": str(exc)})

    def _start_recording(self, ws) -> None:
        def on_result(keys: list[str] | None, error: str | None) -> None:
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.create_task,
                self._recording_done(ws, keys, error),
            )

        try:
            self._recorder.start(on_result)
        except RuntimeError as exc:
            asyncio.ensure_future(
                ws.send_json({"type": "notify", "level": "error", "text": str(exc)})
            )
            return
        asyncio.ensure_future(ws.send_json({"type": "recording", "active": True}))

    async def _recording_done(self, ws, keys: list[str] | None, error: str | None) -> None:
        payload: dict = {"type": "recording", "active": False}
        if keys is not None:
            payload["keys"] = keys
        if error:
            payload["error"] = error
        await ws.send_json(payload)

    async def _after_change(self, ws) -> None:
        await self._send_state(ws)
        await self._reload_config()
