"""Camera: CAP_DSHOW + MJPG + dedicated drain thread (latest frame only).

Proven facts baked in (confirmed by research):
- OpenCV CAP_DSHOW queues frames internally; a slow consumer reads STALE
  frames and lag accumulates (the old 2-3s bug). A dedicated drain thread
  that continuously reads and keeps only the newest frame removes it.
- MJPG fourcc enables 30fps at 720p (default YUY2 caps far lower on many
  webcams).
- BUFFERSIZE=1 where the backend honors it.
- read() never resets to None: repeated delivery is harmless, a dropped
  frame stalls fps (old lesson).
"""

from __future__ import annotations

import threading
import time

import cv2


class Camera:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
        open_retries: int = 5,
    ) -> None:
        self._mirror = mirror
        self._cap = self._open(index, width, height, fps, open_retries)
        self._lock = threading.Lock()
        self._latest: object | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="camera-drain", daemon=True)
        self._thread.start()

    @staticmethod
    def _open(index: int, width: int, height: int, fps: int, retries: int):
        last_err: Exception = RuntimeError("camera open failed")
        for attempt in range(1, retries + 1):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
            last_err = RuntimeError(f"camera {index} not opened (attempt {attempt}/{retries})")
            time.sleep(1.0)
        raise last_err

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception:  # noqa: BLE001 - driver hiccups must not kill the drain
                ok, frame = False, None
            if ok and frame is not None:
                if self._mirror:
                    frame = cv2.flip(frame, 1)
                with self._lock:
                    self._latest = frame
            else:
                self._stop.wait(0.05)

    def read(self) -> object | None:
        """Instant: always returns the newest frame (never None once started)."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._cap.release()
