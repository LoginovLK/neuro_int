"""One Euro Filter: jitter suppression for drawn landmarks.

Casiez et al. 2012: aggressive low-pass when slow (kills sensor grain),
responsive when fast (keeps swipes readable). Same math as the brain's
cursor filter, applied edge-side to the drawn skeleton.
"""

from __future__ import annotations

import math


class LowPassFilter:
    def __init__(self) -> None:
        self._y: float | None = None
        self._s: float | None = None

    def filter(self, value: float, alpha: float) -> float:
        if self._s is None:
            self._s = value
        else:
            self._s = alpha * value + (1.0 - alpha) * self._s
        self._y = value
        return self._s

    def has_value(self) -> bool:
        return self._y is not None


class OneEuroFilter:
    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.2,
        beta: float = 0.03,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = LowPassFilter()
        self._dx = LowPassFilter()
        self._prev_time: float | None = None

    def filter(self, value: float, timestamp_s: float | None = None) -> float:
        if timestamp_s is not None and self._prev_time is not None:
            dt = max(timestamp_s - self._prev_time, 1e-3)
            self.freq = 1.0 / dt
        self._prev_time = timestamp_s

        d_value = 0.0
        if self._x.has_value():
            d_value = (value - self._x._y) * self.freq  # type: ignore[operator]
        ed_value = self._dx.filter(d_value, _alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * abs(ed_value)
        return self._x.filter(value, _alpha(cutoff, self.freq))


class PointFilter2D:
    """Two independent One Euro channels for an (x, y) point."""

    def __init__(self, **kwargs: float) -> None:
        self._fx = OneEuroFilter(**kwargs)
        self._fy = OneEuroFilter(**kwargs)

    def filter(self, x: float, y: float, timestamp_s: float | None = None) -> tuple[float, float]:
        return (
            self._fx.filter(x, timestamp_s),
            self._fy.filter(y, timestamp_s),
        )

    def reset(self) -> None:
        self._fx = OneEuroFilter(**{"min_cutoff": self._fx.min_cutoff, "beta": self._fx.beta})
        self._fy = OneEuroFilter(**{"min_cutoff": self._fy.min_cutoff, "beta": self._fy.beta})


def _alpha(cutoff: float, freq: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)
