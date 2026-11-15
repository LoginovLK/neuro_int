"""Debug window drawing: skeleton, bundle point, status."""

from __future__ import annotations

import cv2
import numpy as np

from neuroint_cursor.hands import HAND_CONNECTIONS
from neuroint_cursor.pose import pinch_center

GREEN = (0, 255, 0)
ORANGE = (0, 160, 255)
GRAY = (90, 90, 90)
WHITE = (255, 255, 255)


def draw(
    frame: np.ndarray,
    result: tuple[list[tuple[float, float]], str, float] | None,
    fps: float,
    sens: float,
    active: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    handed = "-"
    if result is not None:
        pts, handed, _score = result
        for a, b in HAND_CONNECTIONS:
            cv2.line(
                frame,
                (int(pts[a][0] * w), int(pts[a][1] * h)),
                (int(pts[b][0] * w), int(pts[b][1] * h)),
                (180, 230, 180),
                1,
            )
        tx, ty = pinch_center(pts)  # the bundle center drives the cursor
        cv2.circle(frame, (int(tx * w), int(ty * h)), 9, GREEN if active else ORANGE, 2)

    status = "PINCH - cursor follows the bundle" if active else "gather fingertips into a bundle"
    line1 = f"fps {fps:4.1f}   sens {sens:4.1f}   hand {handed}"
    line2 = "[+/-] sens   [q] quit"
    cv2.putText(frame, line1, (10, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, status, (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GREEN if active else ORANGE, 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)
    return frame
