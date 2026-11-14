"""Latest-frame capture wrapper.

Wraps any FrameSource with a dedicated reader thread that continuously
drains the camera. Callers get the most recent frame instantly instead of
a stale one from the driver's internal buffer — this is what removes the
2-3 second video lag (OpenCV/CAP_DSHOW queues frames internally, and a
blocking read() hands them out one by one).
"""

from __future__ import annotations

import threading

from neuroint_edge.capture.base import FrameSource


class LatestFrameSource(FrameSource):
    name = "latest"

    def __init__(self, inner: FrameSource) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._latest: object | None = None
        self._stop = threading.Event()
        self.name = f"latest({inner.name})"
        self._thread = threading.Thread(target=self._loop, name="camera-drain", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._inner.read()
            except Exception:  # noqa: BLE001 - driver hiccups must not kill the drain
                frame = None
            if frame is not None:
                with self._lock:
                    self._latest = frame
            else:
                self._stop.wait(0.05)

    def read(self) -> object | None:
        """Instant: always returns the newest frame (never resets to None —
        repeated delivery is harmless, a dropped frame would stall fps)."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._inner.close()
