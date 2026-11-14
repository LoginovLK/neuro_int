"""Tracked ROI: the next landmark-crop rect from the previous frame's landmarks.

Tracking (like MediaPipe and the PINTO repo's use_tracking) is what keeps
the bones from sliding: a palm box re-detected from scratch every frame
jitters slightly, and the jittered crop makes the landmarks slide. With
tracking the palm detector only runs to (re)acquire a lost hand.

Pure math, no heavy imports — unit-testable without onnxruntime.
"""

from __future__ import annotations

from math import atan2, cos, degrees, floor, hypot, pi, sin

# Tracked-ROI size: hand-bbox diagonal scaled by this. Too small crops
# tank the landmark score; too big just wastes resolution.
TRACK_SIZE_FACTOR = 1.8
# Missed frames before tracking state is cleared (then palm re-acquires).
TRACK_MISS_LIMIT = 10
# A real hand scores 0.6-0.95 on the landmark head; a tracked false
# positive (face, keyboard edges) usually scores lower. Holding a track
# needs the higher bar, so junk tracks die quickly and palm re-acquires.
TRACK_SCORE_MIN = 0.4
# Palm sees nothing anywhere while a track lives -> phantom track, dies
# after this many consecutive empty palm frames.
TRACK_EMPTY_PALM_LIMIT = 2
# Per-frame growth limits for the tracked ROI: a real hand cannot double
# in size or teleport half a frame in 33ms. (Compounding is bounded again
# in hands.py over a 1.5s window — per-frame limits alone still allow
# 1.15^60 runaway growth on hallucinated landmarks.)
TRACK_GROW_MAX = 1.15
TRACK_SHRINK_MIN = 0.7
TRACK_CENTER_JUMP = 0.8  # fraction of the previous ROI size


def normalize_radians(angle: float) -> float:
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))


def rect_from_landmarks(
    points_norm: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
    prev_dir: tuple[float, float] | None,
    prev_rect: list[float] | None = None,
) -> tuple[list[float] | None, tuple[float, float]]:
    """Build the next tracked ROI from the previous frame's landmarks.

    Rotation follows the wrist -> middle-finger-MCP axis (the same
    convention the palm detector uses), smoothed as a direction vector
    (no angle wrap issues). With prev_rect given, size and center are
    limited per frame so garbage input cannot explode the crop.
    Returns (rect | None, smoothed direction).
    """
    px = [(x * frame_w, y * frame_h) for x, y in points_norm]
    wrist, mid = px[0], px[9]
    dx, dy = mid[0] - wrist[0], mid[1] - wrist[1]
    if abs(dx) + abs(dy) < 1e-6:
        return None, (0.0, 1.0) if prev_dir is None else prev_dir
    rotation = normalize_radians(0.5 * pi - atan2(-dy, dx))
    d = (sin(rotation), cos(rotation))
    if prev_dir is not None:
        sx = 0.7 * prev_dir[0] + 0.3 * d[0]
        cy_dir = 0.7 * prev_dir[1] + 0.3 * d[1]
        n = hypot(sx, cy_dir)
        if n > 1e-6:
            d = (sx / n, cy_dir / n)
    rotation = atan2(d[0], d[1])

    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    diag = hypot(max(xs) - min(xs), max(ys) - min(ys))
    size = min(0.9 * min(frame_w, frame_h), max(48.0, TRACK_SIZE_FACTOR * diag))
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    if prev_rect is not None:
        psize = float(prev_rect[2])
        size = min(max(size, TRACK_SHRINK_MIN * psize), TRACK_GROW_MAX * psize)
        jump = TRACK_CENTER_JUMP * psize
        ddx, ddy = cx - prev_rect[0], cy - prev_rect[1]
        dist = hypot(ddx, ddy)
        if dist > jump > 0:
            cx = prev_rect[0] + ddx * jump / dist
            cy = prev_rect[1] + ddy * jump / dist

    return [int(cx), int(cy), int(size), int(size), degrees(rotation)], d
