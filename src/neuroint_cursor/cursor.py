"""Cursor actuator: mapping + One Euro + median-3 + pynput (SetCursorPos).

DPI awareness is set before reading screen metrics so coordinates match
physical pixels on scaled displays (otherwise Windows virtualizes them
and the cursor lands off-target).
"""

from __future__ import annotations

import sys
from collections import deque
from math import hypot
from statistics import median

from pynput.mouse import Button, Controller

from neuroint_cursor.smoothing import PointFilter2D

MIN_SENS = 0.5
MAX_SENS = 8.0

# Pointer acceleration (the OS-mouse answer to the precision-vs-speed
# dilemma). The speed signal is the EMA of the SIGNED velocity vector:
# physiological tremor is oscillatory (its vector average collapses to
# ~0) while a deliberate slow aim is persistent (full magnitude) — so
# the curve can suppress tremor hard without suppressing aim. Speeds
# are in base-gain screen px/s.
G_MIN = 0.08
# freeze band: only NEAR-ZERO speed is suppressed (tremor); deliberate
# aiming moves 1:1 from ~50 px/s up — an aggressive low curve choked the
# aiming band and read as jerk (undershoot-overshoot hunting)
S_FREEZE = 50.0
S_HIGH = 600.0
ACCEL_CAP = 3.0
VEL_EMA = 0.15
# movement pauses longer than this (scroll mode, tracking gap) rebase the
# reference instead of integrating the whole travel in one frame (the
# acceleration curve would boost that leap up to ACCEL_CAP)
REBASE_GAP_S = 0.2
# single-frame outlier gate: a hand jump beyond this fraction of the
# frame in ONE frame (~6 frame-widths/s) is a detector glitch (tracking
# re-acquire, hand ID switch), not human motion — the frame is skipped
# and the cursor HOLDS. Incremental movement has no absolute anchor to
# pull the cursor back, so an integrated outlier would teleport it.
OUTLIER_N = 0.20
# hard cap on the cursor step after gain, as a fraction of the screen
# width per frame (~9000 px/s on a 2560 screen): real flicks stay under
# it, anything above slides instead of teleporting (glitches are already
# gated to zero by OUTLIER_N; this is the last-resort bound)
MAX_STEP_FRAC = 0.11


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
    """Relative (touchpad-style) control with pointer acceleration.

    On engage the cursor STAYS where it is; afterwards each frame's hand
    increment moves the cursor from its current spot, scaled by
    sensitivity times the acceleration curve: a still hand freezes the
    cursor on target (tremor suppressed), a flick sweeps the screen.
    """

    def __init__(
        self,
        sensitivity: float = 1.0,
        min_cutoff: float = 1.2,
        beta: float = 0.01,
        actuate: bool = True,
    ) -> None:
        make_dpi_aware()
        self.screen_w, self.screen_h = screen_size()
        self._sensitivity = sensitivity
        # dry mode (--dry / make smoke): the full pipeline runs but the
        # physical mouse is never touched
        self._actuate = actuate
        self._filter = PointFilter2D(min_cutoff=min_cutoff, beta=beta)
        self._mouse = Controller()
        self._max_step_px = MAX_STEP_FRAC * self.screen_w
        # median-of-3 on the filtered target: kills single-frame outliers
        # (the residual finger twitch amplified by the sensitivity gain)
        self._median_win: deque[tuple[int, int]] = deque(maxlen=3)
        # incremental (pointer-acceleration) mode state
        self._anchor_cursor: tuple[float, float] | None = None
        self._prev_f: tuple[float, float] | None = None
        self._prev_t: float | None = None
        self._vel_ema: tuple[float, float] = (0.0, 0.0)
        self._scroll_ref: float | None = None
        self._scroll_acc = 0.0

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    def set_sensitivity(self, value: float) -> None:
        self._sensitivity = min(MAX_SENS, max(MIN_SENS, value))

    def reset(self, anchor_fx: float, anchor_fy: float) -> None:
        """Engage: freeze the cursor at its current spot and bind the
        current control-point position as the movement reference."""
        self._filter.reset()
        self._median_win.clear()
        self._prev_f = (anchor_fx, anchor_fy)
        self._prev_t = None
        self._vel_ema = (0.0, 0.0)
        pos = self._mouse.position
        self._anchor_cursor = (float(pos[0]), float(pos[1]))

    def _gain_mult(self, speed: float) -> float:
        """Acceleration curve: near-zero speed freezes on target (tremor),
        the whole aiming band moves 1:1, flicks boost up to ACCEL_CAP."""
        if speed < S_FREEZE:
            return max(G_MIN, (speed / S_FREEZE) ** 1.3)
        if speed < S_HIGH:
            return 1.0
        return min(1.0 + (speed - S_HIGH) / S_HIGH, ACCEL_CAP)

    def move(self, fx: float, fy: float, now_s: float) -> tuple[int, int]:
        if self._anchor_cursor is None or self._prev_f is None:
            pos = self._mouse.position
            return int(pos[0]), int(pos[1])
        dt = 1.0 / 30.0
        rebase = self._prev_t is not None and (now_s - self._prev_t) > REBASE_GAP_S
        if self._prev_t is not None and not rebase:
            dt = max(now_s - self._prev_t, 1.0 / 120.0)
        self._prev_t = now_s
        # per-frame hand increment at base gain (screen px)
        if rebase:
            inc_x = inc_y = 0.0
            self._vel_ema = (0.0, 0.0)
        else:
            jump_n = hypot(fx - self._prev_f[0], fy - self._prev_f[1])
            if jump_n > OUTLIER_N:
                # detector glitch (not human motion): hold the cursor,
                # follow the reference so the next frame is incremental
                self._vel_ema = (0.0, 0.0)
                inc_x = inc_y = 0.0
            else:
                inc_x = (fx - self._prev_f[0]) * self.screen_w * self._sensitivity
                inc_y = (fy - self._prev_f[1]) * self.screen_h * self._sensitivity
        self._prev_f = (fx, fy)
        # signed-velocity EMA: tremor oscillates (averages out), a
        # deliberate aim is persistent (full magnitude)
        vx, vy = inc_x / dt, inc_y / dt
        self._vel_ema = (
            VEL_EMA * vx + (1.0 - VEL_EMA) * self._vel_ema[0],
            VEL_EMA * vy + (1.0 - VEL_EMA) * self._vel_ema[1],
        )
        g = self._gain_mult(hypot(*self._vel_ema))
        step_x, step_y = inc_x * g, inc_y * g
        mag = hypot(step_x, step_y)
        if mag > self._max_step_px:
            scale = self._max_step_px / mag
            step_x, step_y = step_x * scale, step_y * scale
        cx = min(max(self._anchor_cursor[0] + step_x, 0), self.screen_w - 1)
        cy = min(max(self._anchor_cursor[1] + step_y, 0), self.screen_h - 1)
        self._anchor_cursor = (cx, cy)
        # One Euro + median-of-3: smooth without losing fast strokes
        sx, sy = self._filter.filter(cx, cy, now_s)
        self._median_win.append((int(sx), int(sy)))
        mx = int(median(p[0] for p in self._median_win))
        my = int(median(p[1] for p in self._median_win))
        if self._actuate:
            self._mouse.position = (mx, my)
        return mx, my

    # ---------------- mouse button gestures ----------------

    def press(self, button: Button) -> None:
        """Pinch down: press and HOLD (drag = hold + move)."""
        if self._actuate:
            self._mouse.press(button)

    def release(self, button: Button) -> None:
        if self._actuate:
            self._mouse.release(button)

    def release_all(self) -> None:
        """Safety: drop every tracked button (hand lost / disengage / quit).

        Dry mode never presses, so it also never releases (a release event
        could drop a button the user is holding on the physical mouse).
        """
        if not self._actuate:
            return
        for button in (Button.left, Button.right, Button.middle):
            self._mouse.release(button)

    def scroll_anchor(self, fy: float) -> None:
        """Enter scroll pose: bind the wheel reference to the hand."""
        self._scroll_ref = fy
        self._scroll_acc = 0.0

    def scroll_feed(self, fy: float) -> None:
        """Vertical hand travel -> vertical wheel steps (40px per step).

        Touchpad feel: hand moves down -> content scrolls down.
        """
        if not self._actuate:
            return
        if self._scroll_ref is None:
            self._scroll_ref = fy
            return
        self._scroll_acc += fy - self._scroll_ref
        self._scroll_ref = fy
        steps = int(self._scroll_acc * self.screen_h / 40.0)
        if steps != 0:
            self._scroll_acc -= steps * 40.0 / self.screen_h
            self._mouse.scroll(0, -steps)

    def scroll_end(self) -> None:
        self._scroll_ref = None
        self._scroll_acc = 0.0
