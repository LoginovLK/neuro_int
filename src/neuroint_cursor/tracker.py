"""Rigid template tracking of a fixed hand pose (phase correlation).

The idea: while the hand holds its pose, the 21 landmarks are a FROZEN
shape that only TRANSLATES. A per-frame phase correlation between the
reference patch and the current patch measures the hand's translation —
sub-pixel, ~0.5ms, no CNN, no per-frame landmark noise, so the skeleton
cannot jitter. The expensive landmark inference re-fits only on
triggers: every REFIT_EVERY frames (drift correction) or immediately
when the correlation response drops (pose change, blur, occlusion).

cv2.phaseCorrelate(ref, cur) returns the CONTENT displacement of cur
relative to ref — add it to the tracked center (verified on synthetic
shifts).
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_PATCH_HALF = 28
MAX_PATCH_HALF = 90


class HandTemplateTracker:
    """Tracks one frozen hand pose by patch translation."""

    def __init__(self, min_response: float = 0.12) -> None:
        self.min_response = min_response
        self._ref: np.ndarray | None = None  # reference patch, float32 gray
        self._half = 0
        self._hann: np.ndarray | None = None
        self.center: tuple[float, float] = (0.0, 0.0)
        self.points: list[tuple[float, float]] = []  # normalized
        self.handedness = "-"
        self.score = 0.0
        self.frame_wh = (0, 0)
        self.age = 0  # frames since the last reference capture

    def capture(
        self,
        frame_gray: np.ndarray,
        center: tuple[float, float],
        rect_size: float,
        points_norm: list[tuple[float, float]],
        handedness: str,
        score: float,
    ) -> None:
        """(Re)capture the reference patch — a re-fit to the current pose."""
        h, w = frame_gray.shape[:2]
        half = int(max(MIN_PATCH_HALF, min(MAX_PATCH_HALF, rect_size * 0.3)))
        cx, cy = int(center[0]), int(center[1])
        x0, y0 = cx - half, cy - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
            return  # patch would leave the frame; keep the old reference
        patch = frame_gray[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
        self._ref = patch.astype(np.float32)
        self._half = half
        self._hann = cv2.createHanningWindow((2 * half, 2 * half), cv2.CV_32F)
        self.center = (float(center[0]), float(center[1]))
        self.points = list(points_norm)
        self.handedness = handedness
        self.score = score
        self.frame_wh = (w, h)
        self.age = 0

    @property
    def active(self) -> bool:
        return self._ref is not None

    def update(
        self, frame_gray: np.ndarray
    ) -> tuple[list[tuple[float, float]], str, float, tuple[float, float]] | None:
        """Translate the frozen pose by the measured patch shift.

        Returns (points, handedness, score, new_center) or None when the
        correlation is too weak (pose change, blur, occlusion, edge).
        """
        if self._ref is None:
            return None
        h, w = frame_gray.shape[:2]
        half = self._half
        cx, cy = int(round(self.center[0])), int(round(self.center[1]))
        x0, y0 = cx - half, cy - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
            return None  # hand drifted to the frame edge
        patch = frame_gray[y0 : y0 + 2 * half, x0 : x0 + 2 * half].astype(np.float32)
        shift, response = cv2.phaseCorrelate(self._ref, patch, self._hann)
        if response < self.min_response:
            return None
        self.center = (self.center[0] + shift[0], self.center[1] + shift[1])
        self.age += 1
        dx_norm, dy_norm = shift[0] / w, shift[1] / h
        points = [(x + dx_norm, y + dy_norm) for x, y in self.points]
        self.points = points
        return points, self.handedness, self.score, self.center
