"""PS3 Eye capture via pseyepy (libusb user-mode driver).

Why this backend exists: the PS3 Eye under Windows auto-exposure produces
heavy green-tinted gain noise (its VGA sensor pushes gain in dark scenes).
Manual gain/exposure + 60fps capture is the fix.
"""

from __future__ import annotations

import cv2
import numpy as np

from neuroint_edge.capture.base import FrameSource


class PseyeSource(FrameSource):
    """pseyepy backend. pseyepy returns RGB; we normalize to BGR for OpenCV."""

    name = "pseyepy"

    def __init__(
        self,
        index: int = 0,
        fps: int = 30,
        gain: int = -1,
        exposure: int = -1,
    ) -> None:
        from pseyepy import Camera  # deferred: optional dependency

        kwargs: dict = {
            "fps": fps,
            "resolution": Camera.RES_LARGE,
            "colour": True,
        }
        if gain >= 0:
            kwargs["gain"] = gain
        if exposure >= 0:
            kwargs["exposure"] = exposure
        self._camera = Camera(index, **kwargs)

    def read(self) -> np.ndarray | None:
        frame, _timestamp = self._camera.read()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @property
    def sensor(self):
        """Direct pseyepy Camera (mutable gain/exposure/whitebalance)."""
        return self._camera

    def close(self) -> None:
        self._camera.end()
