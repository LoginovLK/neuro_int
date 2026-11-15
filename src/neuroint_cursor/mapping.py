"""Detection -> cursor coordinates: the whole frame maps to the whole screen.

No active region: the fingertip anywhere in the camera frame points
anywhere on the screen. Sensitivity is the GAIN around the frame center
(sens 1.0: full frame -> full screen; sens 2.0: half the hand travel
sweeps the whole screen; edges stay reachable at any sensitivity).
Pure math, no I/O — unit-testable.
"""

from __future__ import annotations

MIN_SENS = 0.4
MAX_SENS = 4.0


class CursorMapper:
    def __init__(self, screen_w: int, screen_h: int, sensitivity: float = 1.0) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.sensitivity = sensitivity

    def map(self, fx: float, fy: float) -> tuple[int, int]:
        """Normalized fingertip -> screen pixels. Center-anchored gain."""
        g = min(MAX_SENS, max(MIN_SENS, self.sensitivity))
        nx = min(max(0.5 + (fx - 0.5) * g, 0.0), 1.0)
        ny = min(max(0.5 + (fy - 0.5) * g, 0.0), 1.0)
        return min(int(nx * self.screen_w), self.screen_w - 1), min(
            int(ny * self.screen_h), self.screen_h - 1
        )
