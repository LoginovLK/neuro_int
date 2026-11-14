"""Generic webcam capture via OpenCV.

Fallback backend: any ordinary USB webcam (also a PS3 Eye equipped with a
DirectShow filter driver instead of plain libusb).

Backend order on Windows: DirectShow first (fast startup), then MSMF
(Media Foundation — more resilient on Win11 for modern UVC cameras), then
the platform default. The first backend that opens wins.
"""

from __future__ import annotations

import cv2
import numpy as np

from neuroint_edge.capture.base import FrameSource

_BACKENDS = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)


class OpenCvSource(FrameSource):
    name = "opencv"

    def __init__(self, index: int = 0, fps: int = 30, width: int = 640, height: int = 480) -> None:
        self._cap = None
        last_error = "no backend attempted"
        for backend in _BACKENDS:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                self._cap = cap
                self._backend_name = {
                    cv2.CAP_DSHOW: "dshow",
                    cv2.CAP_MSMF: "msmf",
                }.get(backend, "auto")
                break
            cap.release()
            last_error = f"backend {backend} could not open camera index {index}"
        if self._cap is None:
            raise RuntimeError(f"OpenCV cannot open camera index {index}: {last_error}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()
