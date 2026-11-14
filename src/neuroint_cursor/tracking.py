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
# Cross-check the tracked hand against the palm detector every N frames:
# if the palm detector sees NO palm anywhere while a track is alive, the
# track is a false positive (shelf junk, face) and dies after K checks.
TRACK_RECHECK_EVERY = 12
TRACK_EMPTY_PALM_LIMIT = 2


def normalize_radians(angle: float) -> float:
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))


def rect_from_landmarks(
    points_norm: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
    prev_dir: tuple[float, float] | None,
) -> tuple[list[float] | None, tuple[float, float]]:
    """Build the next tracked ROI from the previous frame's landmarks.

    Rotation follows the wrist -> middle-finger-MCP axis (the same
    convention the palm detector uses), smoothed as a direction vector
    (no angle wrap issues). Returns (rect | None, smoothed direction).
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
    size = int(min(0.9 * min(frame_w, frame_h), max(48.0, TRACK_SIZE_FACTOR * diag)))
    cx = int((max(xs) + min(xs)) / 2)
    cy = int((max(ys) + min(ys)) / 2)
    return [cx, cy, size, size, degrees(rotation)], d
