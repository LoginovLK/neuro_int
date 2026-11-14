"""Edge WebSocket client: GPU hand detection + brain protocol.

The pipeline per frame:
  capture -> auto-exposure -> mirror -> enhance -> skeleton draw
          -> ONNX hand detection (DirectML, GPU) -> LandmarksMsg -> brain

The brain replies with the FSM echo (cursor_active + steering coords), the
edge drives the mouse and relays the echo to the live UI. Camera frames
never leave the process: the MJPEG stream is annotated locally.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time

import cv2
import websockets
from neuroint_protocol.messages import (
    ActionsMsg,
    CommandMsg,
    HelloMsg,
    LandmarksMsg,
    StatusMsg,
    encode_message,
    parse_message,
)

from neuroint_edge.actuator import KeyActuator, ScreenMapper
from neuroint_edge.capture.base import FrameSource
from neuroint_edge.capture.factory import create_capture
from neuroint_edge.capture.quality import FrameEnhancer
from neuroint_edge.capture.usb_reset import restart_camera_devices
from neuroint_edge.config import EdgeSettings, persist_env
from neuroint_edge.ui.hub import UiHub
from neuroint_edge.vision.onnx_hands import OnnxHands
from neuroint_edge.vision.skeleton import draw_skeleton

logger = logging.getLogger(__name__)

_RECONNECT_DELAY_S = 2.0
_READ_TIMEOUT_S = 2.5
_READ_FAILURES_BEFORE_REBUILD = 3


class EdgeClient:
    def __init__(
        self,
        settings: EdgeSettings,
        source: FrameSource,
        hub: UiHub | None = None,
        actuator: KeyActuator | None = None,
        enhancer: FrameEnhancer | None = None,
        hands_detector: OnnxHands | None = None,
    ) -> None:
        self._settings = settings
        self._source = source
        self._hub = hub
        self._actuator = actuator or KeyActuator()
        self._enhancer = enhancer if settings.enhance else None
        self._detector = hands_detector
        self._screen_mapper = ScreenMapper()
        self._brain_ws = None
        self._brain_paused = False
        self._pending_pause = False
        self._last_cursor_active = False
        self._cursor_prev: tuple[float, float] | None = None
        self._auto_exp = self._build_auto_exposure()
        # Camera reads must never block the event loop: dedicated worker.
        self._read_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="camera-read"
        )
        self._detect_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpu-detect"
        )
        self._read_failures = 0
        self._rebuild_failures = 0
        self._usb_restarted = False

    def _build_auto_exposure(self):
        """Zero-touch camera calibration (pseyepy only, when not pinned)."""
        s = self._settings
        if not s.auto_exposure or s.gain >= 0 or s.exposure >= 0:
            return None
        sensor = getattr(self._source, "sensor", None)
        if sensor is None:
            return None
        from neuroint_edge.capture.autoexposure import AutoExposureController

        controller = AutoExposureController(sensor)
        logger.info("auto exposure calibration enabled (raw frames until settled)")
        return controller

    async def send_reload_config(self) -> None:
        """Ask the brain to re-read gestures.yaml (called by the UI editor)."""
        if self._brain_ws is None:
            logger.warning("reload_config: brain not connected")
            return
        await self._brain_ws.send('{"type": "reload_config"}')

    async def send_pause_toggle(self) -> None:
        if self._brain_ws is None:
            # Edge is disconnected: remember the press, apply on reconnect.
            self._pending_pause = not self._pending_pause
            logger.warning(
                "pause_toggle while disconnected — will apply on reconnect (%s)",
                "queued" if self._pending_pause else "cleared",
            )
            return
        await self._brain_ws.send('{"type": "command", "action": "pause_system"}')

    async def run(self) -> None:
        while True:
            try:
                # Generous keepalive: detection bursts can delay pong answers
                # on the brain side; the default 20s timeout misfires.
                async with websockets.connect(
                    self._settings.ws_url,
                    ping_interval=20,
                    ping_timeout=120,
                ) as ws:
                    self._brain_ws = ws
                    logger.info("connected to brain at %s", self._settings.ws_url)
                    await ws.send(
                        HelloMsg(
                            client="edge",
                            capture_fps=self._settings.capture_fps,
                            jpeg_quality=self._settings.jpeg_quality,
                        ).model_dump_json()
                    )
                    if self._pending_pause:
                        self._pending_pause = False
                        await ws.send('{"type": "command", "action": "pause_system"}')
                        logger.info("pending pause toggle applied on reconnect")
                    recv_task = asyncio.create_task(self._recv_loop(ws))
                    try:
                        await self._capture_loop(ws)
                    finally:
                        recv_task.cancel()
            except (OSError, websockets.WebSocketException) as exc:
                logger.warning("brain unavailable (%s), retrying in %.0fs", exc, _RECONNECT_DELAY_S)
                self._brain_ws = None
                await asyncio.sleep(_RECONNECT_DELAY_S)

    async def _capture_loop(self, ws) -> None:
        interval = 1.0 / max(self._settings.capture_fps, 1)
        loop = asyncio.get_running_loop()
        while True:
            frame = await self._read_frame(loop)
            if frame is None:
                continue

            if self._auto_exp is not None:
                # Measure the RAW sensor frame, before any processing.
                self._auto_exp.step(frame)

            if self._settings.mirror:
                frame = cv2.flip(frame, 1)
            if self._enhancer is not None:
                frame = self._enhancer.enhance(frame)

            # GPU hand detection in the worker pool; never block the loop.
            hands: list = []
            if self._detector is not None:
                try:
                    hands = await asyncio.wait_for(
                        loop.run_in_executor(self._detect_pool, self._detector.detect, frame),
                        timeout=2.0,
                    )
                except TimeoutError:
                    logger.warning("hand detection timeout (frame dropped)")
                except Exception as exc:  # noqa: BLE001 - keep the stream alive
                    logger.warning("hand detection error: %s (frame dropped)", exc)

            points, handedness = [], "Right"
            if hands:
                points, handedness = hands[0]

            # Skeleton drawn on the exact frame the detector ran on.
            frame = draw_skeleton(frame, points, self._last_cursor_active)

            # Tell the brain (1 KB of landmarks instead of a 300 KB JPEG).
            await ws.send(
                encode_message(
                    LandmarksMsg(
                        timestamp_ms=int(time.monotonic() * 1000),
                        points=[[x, y] for x, y in points],
                        handedness=handedness,
                    )
                )
            )

            ok, jpeg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._settings.jpeg_quality]
            )
            if ok and self._hub is not None:
                self._hub.publish_frame(jpeg.tobytes())
            await asyncio.sleep(interval)

    async def _read_frame(self, loop):
        """Read one frame WITHOUT blocking the event loop.

        Camera .read() is a blocking OS call; on flaky USB hardware it can
        hang forever. Run it in a worker thread with a timeout, and rebuild
        the capture backend after repeated failures — a hung loop here froze
        the whole edge (UI server included).
        """
        try:
            frame = await asyncio.wait_for(
                loop.run_in_executor(self._read_pool, self._source.read),
                timeout=_READ_TIMEOUT_S,
            )
        except TimeoutError:
            self._read_failures += 1
            logger.warning(
                "camera read timeout (%d/%d) — backend %s may be wedged",
                self._read_failures,
                _READ_FAILURES_BEFORE_REBUILD,
                self._source.name,
            )
            if self._read_failures >= _READ_FAILURES_BEFORE_REBUILD:
                await self._rebuild_capture()
            return None
        except Exception as exc:  # noqa: BLE001 - driver errors happen
            self._read_failures += 1
            logger.warning("camera read error (%d): %s", self._read_failures, exc)
            return None
        self._read_failures = 0
        if frame is None:
            logger.warning("camera returned no frame")
            await asyncio.sleep(0.2)
        return frame

    async def _rebuild_capture(self) -> None:
        logger.warning("rebuilding capture backend (%s)", self._source.name)
        loop = asyncio.get_running_loop()
        old_pool, self._read_pool = (
            self._read_pool,
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-read"),
        )
        try:
            await asyncio.wait_for(loop.run_in_executor(old_pool, self._source.close), timeout=2.0)
        except TimeoutError:
            logger.error("old camera backend hung on close, abandoning it")
        try:
            self._source = await asyncio.wait_for(
                loop.run_in_executor(None, create_capture, self._settings), timeout=8.0
            )
            self._read_failures = 0
            self._cursor_prev = None
            logger.info("capture backend rebuilt: %s", self._source.name)
        except Exception as exc:  # noqa: BLE001 - camera may be wedged at USB level
            logger.error("capture rebuild failed: %s — restarting USB device", exc)
            self._rebuild_failures += 1
            if self._rebuild_failures >= 2 and not self._usb_restarted:
                self._rebuild_failures = 0
                ok = await loop.run_in_executor(None, restart_camera_devices)
                self._usb_restarted = ok
                if ok:
                    logger.info("USB camera restart done, retrying capture")
                else:
                    logger.error(
                        "USB restart unavailable (needs elevated shell): "
                        "replug the camera or reboot the stack"
                    )
            await asyncio.sleep(2.0)

    async def _recv_loop(self, ws) -> None:
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                await self._handle_message(message)
        except websockets.ConnectionClosed:
            logger.error("brain connection lost")
        finally:
            self._brain_ws = None

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = parse_message(raw)
        except ValueError as exc:
            logger.warning("bad message from brain: %s", exc)
            return
        if isinstance(msg, CommandMsg):
            await asyncio.get_running_loop().run_in_executor(None, self._actuator.execute, msg)
        elif isinstance(msg, LandmarksMsg):
            # The brain echoes FSM state back; steer from ITS cursor coords.
            self._last_cursor_active = msg.cursor_active
            if msg.cursor_active and not self._brain_paused:
                await self._steer_cursor(msg.cursor_x, msg.cursor_y)
            else:
                self._cursor_prev = None
            if self._hub is not None:
                await self._hub.broadcast(msg.model_dump())
        elif isinstance(msg, ActionsMsg):
            if self._hub is not None:
                self._hub.set_actions(msg.actions)
        elif isinstance(msg, StatusMsg):
            self._brain_paused = msg.paused
            logger.debug(
                "brain status: fps=%.1f hands=%d paused=%s",
                msg.fps_inference,
                msg.hands_seen,
                msg.paused,
            )
            if self._hub is not None:
                await self._hub.broadcast(msg.model_dump())

    async def _steer_cursor(self, x_norm: float, y_norm: float) -> None:
        """Drive the mouse: relative deltas by default, absolute on request."""
        s = self._settings
        if s.cursor_mode == "relative":
            if self._cursor_prev is None:
                # First frame after entering/losing cursor mode: just anchor.
                self._cursor_prev = (x_norm, y_norm)
                return
            dx = x_norm - self._cursor_prev[0]
            dy = y_norm - self._cursor_prev[1]
            self._cursor_prev = (x_norm, y_norm)
            if dx == 0.0 and dy == 0.0:
                return
            await asyncio.get_running_loop().run_in_executor(
                None,
                self._actuator.move_cursor_relative,
                dx,
                dy,
                s.cursor_sensitivity,
            )
        else:
            await asyncio.get_running_loop().run_in_executor(
                None, self._actuator.move_cursor, x_norm, y_norm, self._screen_mapper
            )

    def set_cursor_prefs(self, sensitivity: float, mode: str) -> dict:
        """Apply cursor prefs at runtime and persist them to .env."""
        if mode not in ("relative", "absolute"):
            raise ValueError(f"unknown cursor mode {mode!r}")
        sensitivity = float(min(max(sensitivity, 0.3), 8.0))
        self._settings.cursor_sensitivity = sensitivity
        self._settings.cursor_mode = mode
        persist_env("NEUROINT_CURSOR_SENSITIVITY", str(sensitivity))
        persist_env("NEUROINT_CURSOR_MODE", mode)
        logger.info("cursor prefs: mode=%s sens=%.2f", mode, sensitivity)
        return {"sensitivity": sensitivity, "mode": mode}
