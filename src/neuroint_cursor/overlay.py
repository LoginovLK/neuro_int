"""Debug window drawing: skeleton, control point, pinch state, status."""

from __future__ import annotations

import cv2
import numpy as np

# MediaPipe hand skeleton edges (wrist->thumb->fingers->pinky->wrist).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_TIP = 12

GREEN = (0, 255, 0)
ORANGE = (0, 160, 255)
BLUE = (255, 160, 0)
GRAY = (90, 90, 90)
WHITE = (255, 255, 255)


def draw(
    frame: np.ndarray,
    result: tuple[list[tuple[float, float]], str, float] | None,
    fps: float,
    sens: float,
    status: str,
    active: bool = False,
    lmb: bool = False,
    rmb: bool = False,
    scroll: bool = False,
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
        # control point: the index knuckle drives the cursor (it stays put
        # while the fingertips pinch, so clicks land where the cursor is)
        cx, cy = int(pts[INDEX_MCP][0] * w), int(pts[INDEX_MCP][1] * h)
        cv2.circle(frame, (cx, cy), 9, GREEN if active else ORANGE, 2)
        # pinch indicators: thumb->index (LMB) and thumb->middle (RMB)
        tt = (int(pts[THUMB_TIP][0] * w), int(pts[THUMB_TIP][1] * h))
        it = (int(pts[INDEX_TIP][0] * w), int(pts[INDEX_TIP][1] * h))
        mt = (int(pts[MIDDLE_TIP][0] * w), int(pts[MIDDLE_TIP][1] * h))
        cv2.line(frame, tt, it, GREEN if lmb else GRAY, 3 if lmb else 1)
        cv2.line(frame, tt, mt, BLUE if rmb else (60, 60, 60), 3 if rmb else 1)

    line1 = f"fps {fps:4.1f}   sens {sens:4.1f}   hand {handed}"
    line2 = "[+/-] sens   [q] quit"
    cv2.putText(frame, line1, (10, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, status, (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GREEN if active or scroll else ORANGE, 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)
    return frame
