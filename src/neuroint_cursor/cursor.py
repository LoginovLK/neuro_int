"""Cursor actuator: mapping + One Euro + median-3 + pynput (SetCursorPos).

DPI awareness is set before reading screen metrics so coordinates match
physical pixels on scaled displays (otherwise Windows virtualizes them
and the cursor lands off-target).
"""

from __future__ import annotations

import sys
from collections import deque
from statistics import median

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
    """Relative (touchpad-style) control.

    On engage (reset with the bundle anchor) the cursor STAYS where it
    is; afterwards the hand DELTA from the engage anchor drives the
    cursor from that spot, scaled by sensitivity - no jump to wherever
    the hand happens to be.
    """

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
        # median-of-3 on the filtered target: kills single-frame outliers
        # (the residual finger twitch amplified by the sensitivity gain)
        self._median_win: deque[tuple[int, int]] = deque(maxlen=3)
        self._anchor_frame: tuple[float, float] | None = None
        self._anchor_cursor: tuple[int, int] | None = None

    @property
    def sensitivity(self) -> float:
        return self.mapper.sensitivity

    def set_sensitivity(self, value: float) -> None:
        self.mapper.sensitivity = min(8.0, max(0.5, value))

    def reset(self, anchor_fx: float, anchor_fy: float) -> None:
        """Engage: freeze the cursor at its current spot and bind the
        current bundle position as the hand's zero point."""
        self._filter.reset()
        self._last = None
        self._median_win.clear()
        self._anchor_frame = (anchor_fx, anchor_fy)
        pos = self._mouse.position
        self._anchor_cursor = (int(pos[0]), int(pos[1]))

    def center(self) -> None:
        """The zero gesture: park the cursor in the middle of the screen."""
        self._anchor_cursor = (self.screen_w // 2, self.screen_h // 2)
        self._mouse.position = self._anchor_cursor
        self._filter.reset()
        self._last = None
        self._median_win.clear()

    def move(self, fx: float, fy: float, now_s: float) -> tuple[int, int]:
        if self._anchor_frame is None or self._anchor_cursor is None:
            pos = self._mouse.position
            return int(pos[0]), int(pos[1])
        dx = (fx - self._anchor_frame[0]) * self.screen_w * self.mapper.sensitivity
        dy = (fy - self._anchor_frame[1]) * self.screen_h * self.mapper.sensitivity
        sx = min(max(self._anchor_cursor[0] + dx, 0), self.screen_w - 1)
        sy = min(max(self._anchor_cursor[1] + dy, 0), self.screen_h - 1)
        # One Euro + median-of-3: smooth without losing fast strokes
        sx, sy = self._filter.filter(sx, sy, now_s)
        self._last = (sx, sy)
        self._median_win.append((int(sx), int(sy)))
        mx = int(median(p[0] for p in self._median_win))
        my = int(median(p[1] for p in self._median_win))
        self._mouse.position = (mx, my)
        return mx, my
