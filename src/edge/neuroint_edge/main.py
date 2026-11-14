"""Edge entrypoint: camera -> GPU hand detection -> brain -> keyboard/mouse.

Services started together:
  - EdgeClient: WS to the brain (landmarks out, commands in, cursor steering)
  - UiServer:   loopback HTTP (MJPEG video + static assets) and WS (editor)
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import sys
import time

from neuroint_edge import __version__
from neuroint_edge.capture import create_capture
from neuroint_edge.capture.quality import FrameEnhancer
from neuroint_edge.client import EdgeClient
from neuroint_edge.config import settings
from neuroint_edge.gesture_store import GestureStore
from neuroint_edge.key_recorder import ComboRecorder
from neuroint_edge.ui import UiHub, UiServer
from neuroint_edge.vision.onnx_hands import OnnxHands


def _setup_logging() -> logging.Logger:
    stream = sys.stdout
    if stream is None:
        # pythonw has no console; fall back to a log file so background
        # runs (make edge-start) stay debuggable.
        pathlib.Path("logs").mkdir(exist_ok=True)
        stream = open("logs/edge.log", "a", encoding="utf-8", buffering=1)
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=stream,
    )
    return logging.getLogger(__name__)


def _open_capture(logger: logging.Logger):
    """Open the camera with retries and a USB device reset in between.

    A hung USB webcam can refuse to reopen for minutes; failing hard here
    would take the whole edge down before the UI ever starts.
    """
    for attempt in range(1, 6):
        try:
            source = create_capture(settings)
            logger.info("capture backend ready (%s) on attempt %d", source.name, attempt)
            return source
        except Exception as exc:  # noqa: BLE001 - driver level failures happen
            logger.warning("camera open failed (attempt %d/5): %s", attempt, exc)
            if attempt == 3:
                from neuroint_edge.capture.usb_reset import restart_camera_devices

                if restart_camera_devices():
                    logger.info("USB camera devices restarted, retrying")
            time.sleep(3.0)
    raise RuntimeError("camera could not be opened after 5 attempts (see logs above)")


def main() -> None:
    logger = _setup_logging()
    if sys.platform == "win32":
        # Proactor loop spams ConnectionResetError tracebacks whenever a
        # browser tab refreshes; the selector loop does not.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logger.info("neuroint edge %s starting", __version__)

    source = _open_capture(logger)
    hands_detector = OnnxHands(
        palm_model=settings.vision_onnx_palm,
        landmark_model=settings.vision_onnx_landmark,
        score_threshold=settings.vision_score_threshold,
        class_score_threshold=settings.vision_class_threshold,
        mirror=settings.mirror,
    )
    hub = UiHub()
    store = GestureStore(settings.gestures_config)
    recorder = ComboRecorder()
    client = EdgeClient(
        settings, source, hub=hub, enhancer=FrameEnhancer(), hands_detector=hands_detector
    )

    def get_cursor_prefs() -> dict:
        return {
            "sensitivity": settings.cursor_sensitivity,
            "mode": settings.cursor_mode,
        }

    ui = UiServer(
        hub=hub,
        store=store,
        recorder=recorder,
        reload_config=client.send_reload_config,
        pause_toggle=client.send_pause_toggle,
        set_cursor_prefs=client.set_cursor_prefs,
        cursor_prefs=get_cursor_prefs,
        port=settings.ui_port,
    )

    async def run_all() -> None:
        tasks = [client.run()]
        if settings.ui_enabled:
            tasks.append(ui.run())
        await asyncio.gather(*tasks)

    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        logger.info("stopping edge")
    finally:
        source.close()


if __name__ == "__main__":
    main()
