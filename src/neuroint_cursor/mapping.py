"""Detection -> cursor coordinates: absolute pointing over an active region.

The fingertip inside a central active region of the camera frame maps to
the whole screen (like a touchpad, but absolute). Higher sensitivity =
larger active region = the same hand travel sweeps more screen.
Pure math, no I/O — unit-testable.
"""

from __future__ import annotations

MIN_REGION = 0.15
MAX_REGION = 1.0
REGION_PER_SENS = 0.5  # sensitivity 1.0 -> middle half of the frame


class CursorMapper:
    def __init__(self, screen_w: int, screen_h: int, sensitivity: float = 1.0) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.sensitivity = sensitivity

    @property
    def region_fraction(self) -> float:
        return min(MAX_REGION, max(MIN_REGION, REGION_PER_SENS * self.sensitivity))

    def region(self) -> tuple[float, float, float, float]:
        """Active region as normalized (x0, y0, x1, y1), centered."""
        f = self.region_fraction
        off = (1.0 - f) / 2.0
        return (off, off, off + f, off + f)

    def map(self, fx: float, fy: float) -> tuple[int, int]:
        """Normalized fingertip -> screen pixels, clamped to the region."""
        x0, y0, x1, y1 = self.region()
        nx = min(max((fx - x0) / (x1 - x0), 0.0), 1.0)
        ny = min(max((fy - y0) / (y1 - y0), 0.0), 1.0)
        return int(nx * self.screen_w), int(ny * self.screen_h)
