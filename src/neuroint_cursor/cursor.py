"""Cursor actuator: mapping + One Euro + pynput (SetCursorPos).

DPI awareness is set before reading screen metrics so coordinates match
physical pixels on scaled displays (otherwise Windows virtualizes them
and the cursor lands off-target).
"""

from __future__ import annotations

import sys

from pynput.mouse import Controller

from neuroint_cursor.mapping import CursorMapper
from neuroint_cursor.smoothing import PointFilter2D


def make_dpi_aware() -> None:
    if sys.platform == "win32":
        import ctypes
        import logging

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception as exc:  # older Windows: fall back
            logging.getLogger(__name__).debug("SetProcessDpiAwareness unavailable: %s", exc)
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception as exc2:  # nothing else to do
                logging.getLogger(__name__).debug("SetProcessDPIAware unavailable: %s", exc2)


def screen_size() -> tuple[int, int]:
    if sys.platform != "win32":
        raise RuntimeError("windows only")
    import ctypes

    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


class CursorActuator:
    def __init__(
        self,
        sensitivity: float = 1.0,
        min_cutoff: float = 1.2,
        beta: float = 0.01,
    ) -> None:
        make_dpi_aware()
        self.screen_w, self.screen_h = screen_size()
        self._min_cutoff = min_cutoff
        self._beta = beta
        self.mapper = CursorMapper(self.screen_w, self.screen_h, sensitivity)
        self._filter = PointFilter2D(min_cutoff=min_cutoff, beta=beta)
        self._mouse = Controller()
        self._last: tuple[float, float] | None = None

    @property
    def sensitivity(self) -> float:
        return self.mapper.sensitivity

    def set_sensitivity(self, value: float) -> None:
        self.mapper.sensitivity = min(3.0, max(0.5, value))

    def reset(self) -> None:
        """Drop filter history (call when control re-engages)."""
        self._filter.reset()
        self._last = None

    def move(self, fx: float, fy: float, now_s: float) -> tuple[int, int]:
        sx, sy = self.mapper.map(fx, fy)
        # Teleport: a hand re-acquired far from the last filtered position
        # must snap, not glide across the screen (gliding reads as a jump).
        if self._last is not None:
            dist = ((sx - self._last[0]) ** 2 + (sy - self._last[1]) ** 2) ** 0.5
            if dist > 0.35 * min(self.screen_w, self.screen_h):
                self._filter.reset()
                self._last = None
        sx, sy = self._filter.filter(sx, sy, now_s)
        self._last = (sx, sy)
        self._mouse.position = (int(sx), int(sy))
        return int(sx), int(sy)
