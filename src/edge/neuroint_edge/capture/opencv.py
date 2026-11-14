"""Generic webcam capture via OpenCV DirectShow.

Fallback backend: any ordinary USB webcam (also a PS3 Eye equipped with a
DirectShow filter driver instead of plain libusb).
"""

from __future__ import annotations

import cv2
import numpy as np

from neuroint_edge.capture.base import FrameSource


class OpenCvSource(FrameSource):
    name = "opencv"

    def __init__(self, index: int = 0, fps: int = 30, width: int = 640, height: int = 480) -> None:
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open camera index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()
