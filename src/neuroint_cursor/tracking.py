"""Tracked ROI: the next landmark-crop rect from the previous frame's landmarks.

A verbatim port of the MediaPipe hand-tracking rect chain (the industry
pipeline; sources: mediapipe/modules/hand_landmark/calculators/
hand_landmarks_to_rect_calculator.cc and hand_landmark_landmarks_to_roi.pbtxt):

1. HandLandmarksToRectCalculator — the rect comes from the 12 SKELETON
   landmarks only (wrist, thumb chain, index/middle/ring MCP+PIP; NO
   fingertips, they jitter). Rotation: wrist -> mean PIP of index/middle/
   ring. Center and size: bbox of that landmark subset, re-projected into
   the rotated frame.
2. RectTransformationCalculator — shift_y=-0.1 along the rect's own y-axis
   (makes room for the fingers above the PIP line), square_long, scale 2.0.

The palm detector never contributes to the tracked rect: in MediaPipe's
hand_landmark_tracking_cpu.pbtxt the GateCalculator drops the image before
PalmDetection whenever the previous frame already had hands. It only
(re)acquires a lost hand.

Pure math, no heavy imports — unit-testable without onnxruntime.
"""

from __future__ import annotations

from math import atan2, cos, degrees, floor, hypot, pi, sin

# RectTransformationCalculator options from hand_landmark_landmarks_to_roi.pbtxt.
TRACK_SCALE = 2.0
TRACK_SHIFT_Y = -0.1

# Partial-landmark set of HandLandmarksToRectCalculator (kPartialLandmarkIndices).
_PARTIAL = (0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)
WRIST = 0
PIP_INDEX, PIP_MIDDLE, PIP_RING = 6, 10, 14
# Missed frames before tracking state is cleared (then palm re-acquires).
TRACK_MISS_LIMIT = 10
# Tracking needs a lower bar than acquisition: motion blur dips real scores.
TRACK_SCORE_MIN = 0.3


def normalize_radians(angle: float) -> float:
    return angle - 2 * pi * floor((angle - (-pi)) / (2 * pi))


def rect_from_landmarks(
    points_norm: list[tuple[float, float]], frame_w: int, frame_h: int
) -> list[float] | None:
    """HandLandmarksToRectCalculator + RectTransformationCalculator, in pixels.

    Returns [cx, cy, size, size, rotation_degrees] — the same rect format
    _rotate_and_crop and the crop->frame decode in hands.py use.
    """
    pts = {
        i: (points_norm[i][0] * frame_w, points_norm[i][1] * frame_h) for i in _PARTIAL
    }

    # ComputeRotation: wrist -> weighted mean of index/middle/ring PIP
    # (exactly the C++ averaging: (index+ring)/2, then with middle /2).
    x0, y0 = pts[WRIST]
    x1 = ((pts[PIP_INDEX][0] + pts[PIP_RING][0]) / 2.0 + pts[PIP_MIDDLE][0]) / 2.0
    y1 = ((pts[PIP_INDEX][1] + pts[PIP_RING][1]) / 2.0 + pts[PIP_MIDDLE][1]) / 2.0
    dx, dy = x1 - x0, y1 - y0
    if hypot(dx, dy) < 1e-6:
        return None
    rotation = normalize_radians(0.5 * pi - atan2(-dy, dx))
    ca, sa = cos(rotation), sin(rotation)

    # Axis-aligned bbox of the partial landmarks.
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    ax_cx = (max(xs) + min(xs)) / 2.0
    ax_cy = (max(ys) + min(ys)) / 2.0

    # Project the landmarks into the rect frame (rotate by -rotation) and
    # take the bbox there — this is the rect's true width/height/center.
    proj: list[tuple[float, float]] = []
    for i in _PARTIAL:
        ox, oy = pts[i][0] - ax_cx, pts[i][1] - ax_cy
        proj.append((ox * ca + oy * sa, -ox * sa + oy * ca))
    pxs = [p[0] for p in proj]
    pys = [p[1] for p in proj]
    width = max(pxs) - min(pxs)
    height = max(pys) - min(pys)
    proj_cx = (max(pxs) + min(pxs)) / 2.0
    proj_cy = (max(pys) + min(pys)) / 2.0

    # RectTransformationCalculator, in the C++ order: shift first (along the
    # rotated axes, with the pre-scale size), then square_long, then scale.
    shift_x_img = -height * TRACK_SHIFT_Y * sa
    shift_y_img = height * TRACK_SHIFT_Y * ca
    size = max(width, height) * TRACK_SCALE
    cx = ax_cx + proj_cx * ca - proj_cy * sa + shift_x_img
    cy = ax_cy + proj_cx * sa + proj_cy * ca + shift_y_img

    # One guard against a garbage frame: the crop may not exceed the frame
    # (MediaPipe has no cap; a hallucinated giant rect would just tank the
    # landmark score and drop the track).
    size = min(size, 0.95 * min(frame_w, frame_h))
    return [int(cx), int(cy), int(size), int(size), degrees(rotation)]
