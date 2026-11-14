"""Brain WebSocket service: hand landmarks in, FSM state + commands out.

Since the GPU hand detection moved to the edge agent (DirectML on the host
GPU), the brain no longer receives camera frames: it consumes lightweight
LandmarksMsg objects, runs the gesture FSM, maps events to whitelisted
actions and broadcasts commands. The container needs no OpenCV/MediaPipe
at all — it is the decision layer, not the perception layer.
"""

from __future__ import annotations

import asyncio
import logging
import time

import websockets
from neuroint_protocol.gestures import GestureEvent
from neuroint_protocol.messages import (
    ActionsMsg,
    CommandMsg,
    GestureEventMsg,
    HelloMsg,
    LandmarksMsg,
    ReloadConfigMsg,
    StatusMsg,
    encode_message,
    parse_message,
)
from websockets.asyncio.server import ServerConnection

from neuroint_brain.actions import registry
from neuroint_brain.actions.mapper import ActionMapper, load_mapping
from neuroint_brain.config import settings
from neuroint_brain.gestures.fsm import GestureFSM
from neuroint_brain.vision.hands import HandObservation

logger = logging.getLogger(__name__)


class BrainServer:
    def __init__(
        self,
        fsm: GestureFSM,
        mapper: ActionMapper,
        # Container-internal listen address; compose exposes loopback only.
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 8765,
    ) -> None:
        self._fsm = fsm
        self._mapper = mapper
        self._host = host
        self._port = port

        self._paused = False
        self._started_at = time.monotonic()
        self._hands_seen = 0
        self._frame_times: list[float] = []
        self._clients: set[ServerConnection] = set()

    async def serve(self) -> None:
        # Matching keepalive budget for the edge client (see its side):
        # short pings, long timeout, so hiccups never kill the connection.
        async with websockets.serve(
            self._handler,
            self._host,
            self._port,
            ping_interval=20,
            ping_timeout=120,
        ):
            logger.info("brain listening on %s:%d", self._host, self._port)
            await asyncio.Future()  # run until cancelled

    async def _handler(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        await self._send_actions(ws)
        status_task = asyncio.create_task(self._status_loop(ws))
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue  # frames are gone; protocol is text-only now
                await self._handle_text(message)
        except websockets.ConnectionClosed:
            logger.info("edge disconnected")
        finally:
            status_task.cancel()
            self._clients.discard(ws)

    async def _handle_text(self, raw: str) -> None:
        try:
            msg = parse_message(raw)
        except ValueError as exc:
            logger.warning("bad control message: %s", exc)
            return
        if isinstance(msg, HelloMsg):
            logger.info(
                "edge hello: %s (fps=%d, jpeg_q=%d)", msg.client, msg.capture_fps, msg.jpeg_quality
            )
        elif isinstance(msg, LandmarksMsg):
            await self._process_landmarks(msg)
        elif isinstance(msg, ReloadConfigMsg):
            await self._reload_config()
        elif isinstance(msg, CommandMsg) and msg.action == "pause_system":
            # Edge may request a pause toggle too (e.g. from a UI button).
            await self._toggle_pause()

    async def _reload_config(self) -> None:
        try:
            self._mapper = load_mapping(settings.gestures_config)
        except (ValueError, OSError) as exc:
            logger.error("config reload failed: %s", exc)
            await self._send_actions()
            return
        logger.info("gestures.yaml reloaded")
        await self._send_actions()

    async def _send_actions(self, ws: ServerConnection | None = None) -> None:
        msg = ActionsMsg(actions=[spec.as_dict() for spec in registry.list_actions()])
        if ws is not None:
            await ws.send(encode_message(msg))
        else:
            await self._broadcast(msg)

    async def _process_landmarks(self, msg: LandmarksMsg) -> None:
        self._note_fps()
        self._hands_seen = 1 if msg.points else 0
        hand = self._to_observation(msg)
        events = self._fsm.feed(hand)
        await self._broadcast(self._build_landmarks(hand))
        for event in events:
            await self._dispatch(event)

    @staticmethod
    def _to_observation(msg: LandmarksMsg) -> HandObservation | None:
        if not msg.points:
            return None
        ts = msg.timestamp_ms or int(time.monotonic() * 1000)
        return HandObservation(
            landmarks=tuple((p[0], p[1]) for p in msg.points),
            handedness=msg.handedness or "Right",
            timestamp_ms=ts,
        )

    def _build_landmarks(self, hand: HandObservation | None) -> LandmarksMsg:
        fsm = self._fsm
        # Paused = full kill switch: no cursor steering, no state noise.
        # Only the kill gesture itself (mapped to pause_system) still works,
        # because _dispatch always processes pause_system.
        if self._paused:
            return LandmarksMsg(
                points=[[x, y] for x, y in hand.landmarks] if hand else [],
                handedness=hand.handedness if hand else "",
                pose=fsm.last_pose.value,
                fsm_state="paused",
                cursor_active=False,
            )
        return LandmarksMsg(
            points=[[x, y] for x, y in hand.landmarks] if hand else [],
            handedness=hand.handedness if hand else "",
            pose=fsm.last_pose.value,
            fsm_state=fsm.state_name,
            cursor_active=fsm.cursor_active,
            cursor_x=fsm.cursor_x,
            cursor_y=fsm.cursor_y,
        )

    async def _dispatch(self, event: GestureEvent) -> None:
        command = self._mapper.map(event)
        if command is None:
            return
        if command.action == "pause_system":
            await self._toggle_pause()
            return
        if self._paused:
            logger.info("paused: dropping command %s", command.action)
            return
        logger.info("command -> edge: %s (origin=%s)", command.action, command.origin)
        await self._broadcast(
            GestureEventMsg(pose=event.pose.value, direction=event.direction.value)
        )
        await self._broadcast(command)

    async def _toggle_pause(self) -> None:
        self._paused = not self._paused
        logger.info("gesture control %s", "PAUSED" if self._paused else "RESUMED")
        await self._broadcast(StatusMsg(paused=self._paused, fps_inference=self._fps()))

    async def _status_loop(self, ws: ServerConnection) -> None:
        while True:
            await asyncio.sleep(2.0)
            try:
                await ws.send(
                    encode_message(
                        StatusMsg(
                            fps_inference=self._fps(),
                            hands_seen=self._hands_seen,
                            paused=self._paused,
                            uptime_s=round(time.monotonic() - self._started_at, 1),
                        )
                    )
                )
            except websockets.ConnectionClosed:
                return

    async def _broadcast(self, msg) -> None:
        payload = encode_message(msg)
        dead: list[ServerConnection] = []
        for client in self._clients:
            try:
                await client.send(payload)
            except websockets.ConnectionClosed:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    def _note_fps(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        cutoff = now - 5.0
        self._frame_times = [t for t in self._frame_times if t > cutoff]

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return round((len(self._frame_times) - 1) / span, 1) if span > 0 else 0.0
