"""Draw the hand skeleton onto the frame the detector ran on (edge-side).

Same job as brain's version, but here: the frame never leaves the process,
so the skeleton is pixel-perfect by construction.
"""

from __future__ import annotations

import cv2
import numpy as np

# MediaPipe hand topology: index pairs of the 21 landmarks.
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

_COLOR_ACTIVE = (244, 155, 139)  # BGR indigo
_COLOR_TRACKING = (183, 200, 86)  # BGR teal
_TIP_IDS = (4, 8, 12, 16, 20)


def draw_skeleton(
    frame_bgr: np.ndarray,
    points: list[tuple[float, float]] | None,  # normalized
    cursor_active: bool,
) -> np.ndarray:
    if not points:
        return frame_bgr
    color = _COLOR_ACTIVE if cursor_active else _COLOR_TRACKING
    h, w = frame_bgr.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in points]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        radius = 4 if i in _TIP_IDS else 3
        cv2.circle(
            frame_bgr,
            (px, py),
            radius,
            (255, 255, 255) if i in _TIP_IDS else color,
            -1,
            cv2.LINE_AA,
        )
    return frame_bgr
