"""In-memory bridge between the perception pipeline and browser clients.

Holds the latest JPEG frame for the MJPEG stream and the set of connected
WebSocket UI clients for JSON broadcasts. Single-threaded asyncio usage.
"""

from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class UiHub:
    def __init__(self) -> None:
        self._frame: bytes | None = None
        self._clients: set = set()  # aiohttp WebSocketResponse
        self._actions: list[dict] = []
        self._last_status: dict | None = None

    # --- frames (MJPEG) ---

    def publish_frame(self, jpeg: bytes) -> None:
        self._frame = jpeg

    def latest_frame(self) -> bytes | None:
        return self._frame

    # --- last brain status (for instant display on page load) ---

    def set_status(self, status: dict) -> None:
        self._last_status = status

    @property
    def last_status(self) -> dict | None:
        return self._last_status

    # --- websocket clients ---

    def attach(self, ws) -> None:
        self._clients.add(ws)

    def detach(self, ws) -> None:
        self._clients.discard(ws)

    def client_count(self) -> int:
        return len(self._clients)

    def set_actions(self, actions: list[dict]) -> None:
        self._actions = actions

    @property
    def actions(self) -> list[dict]:
        return self._actions

    async def broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        text = json.dumps(payload, separators=(",", ":"))
        dead = []
        for ws in self._clients:
            try:
                await ws.send_str(text)
            except Exception:  # noqa: BLE001 - any send failure drops the client
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def broadcast_threadsafe(self, payload: dict) -> None:
        """Broadcast from a non-asyncio thread (pynput callbacks)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("broadcast_threadsafe called without a running loop")
            return
        loop.call_soon_threadsafe(asyncio.create_task, self.broadcast(payload))
