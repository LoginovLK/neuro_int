"""Normalized hand observation types (framework-free, easily testable)."""

from __future__ import annotations

import dataclasses
import typing

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

Landmark = tuple[float, float]  # (x, y) normalized to frame size


@dataclasses.dataclass(frozen=True)
class HandObservation:
    """One detected hand in normalized image coordinates."""

    landmarks: tuple[Landmark, ...]  # 21 points
    handedness: str  # "Left" / "Right" (MediaPipe's view, mirrored frames swap it)
    timestamp_ms: int  # frame timestamp supplied by the caller

    def palm_size(self) -> float:
        """Distance wrist->middle MCP: the natural scale unit of the hand.

        Swipe thresholds expressed in palm widths stay stable whether the
        hand is near the camera or far away.
        """
        return _distance(self.landmarks[WRIST], self.landmarks[MIDDLE_MCP])

    def fingertip(self) -> Landmark:
        """Index fingertip: the pointer used for swipes."""
        return self.landmarks[INDEX_TIP]

    def finger_extended(self, tip: int, pip: int) -> bool:
        """A finger counts as extended when its tip is farther from the
        wrist than its PIP joint (with a small margin against jitter)."""
        wrist = self.landmarks[WRIST]
        return _distance(self.landmarks[tip], wrist) > _distance(self.landmarks[pip], wrist) * 1.12


def _distance(a: Landmark, b: Landmark) -> float:
    return typing.cast(float, ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
