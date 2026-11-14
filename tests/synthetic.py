"""Synthetic hand landmarks for deterministic gesture tests.

Geometry: wrist at the bottom, palm height ``scale`` above it, fingers
growing upward (y decreases). A finger counts as extended when its tip is
farther from the wrist than its PIP joint (see HandObservation).
"""

from __future__ import annotations

from neuroint_brain.vision.hands import INDEX_TIP, THUMB_TIP, HandObservation, Landmark
from neuroint_protocol.gestures import Pose

SCALE = 0.12  # palm size in normalized units
WRIST_POS: Landmark = (0.5, 0.8)


def _finger_points(mcp: Landmark, extended: bool) -> tuple[Landmark, Landmark]:
    pip = (mcp[0], mcp[1] - 0.45 * SCALE)
    tip_y = mcp[1] - (1.35 * SCALE if extended else 0.15 * SCALE)
    return pip, (mcp[0], tip_y)


def synthetic_hand(
    pose: Pose,
    timestamp_ms: int,
    index_tip_override: Landmark | None = None,
    pinch: bool = False,
) -> HandObservation:
    """Build a full 21-landmark hand in the requested pose.

    ``index_tip_override`` repositions the index fingertip (landmark 8) for
    swipe-trajectory tests; the pose is still classified from the rest.
    ``pinch`` moves the thumb tip next to the index tip (PINCH pose).
    """
    index_mcp: Landmark = (WRIST_POS[0] - 0.35 * SCALE, WRIST_POS[1] - 0.9 * SCALE)
    middle_pos: Landmark = (WRIST_POS[0], WRIST_POS[1] - SCALE)
    ring_mcp: Landmark = (WRIST_POS[0] + 0.35 * SCALE, WRIST_POS[1] - 0.9 * SCALE)
    pinky_mcp: Landmark = (WRIST_POS[0] + 0.7 * SCALE, WRIST_POS[1] - 0.75 * SCALE)

    extended_map = {
        Pose.OPEN_PALM: [True, True, True, True],
        Pose.FIST: [False, False, False, False],
        Pose.INDEX_UP: [True, False, False, False],
        Pose.TWO_UP: [True, True, False, False],
        Pose.PINCH: [True, False, False, False],
        Pose.UNKNOWN: [True, False, True, False],
    }
    if pose not in extended_map:
        raise ValueError(f"synthetic_hand does not support pose {pose}")
    index_ext, middle_ext, ring_ext, pinky_ext = extended_map[pose]

    landmarks: list[Landmark] = [WRIST_POS]
    landmarks += [(WRIST_POS[0] - 0.8 * SCALE, WRIST_POS[1] - 0.3 * SCALE)] * 4  # thumb, unused
    for mcp, ext in (
        (index_mcp, index_ext),
        (middle_pos, middle_ext),
        (ring_mcp, ring_ext),
        (pinky_mcp, pinky_ext),
    ):
        pip, tip = _finger_points(mcp, ext)
        landmarks += [mcp, pip, (pip[0], (pip[1] + tip[1]) / 2), tip]

    if index_tip_override is not None:
        landmarks[INDEX_TIP] = index_tip_override

    if pinch:
        index_tip = landmarks[INDEX_TIP]
        landmarks[THUMB_TIP] = (index_tip[0] + SCALE * 0.06, index_tip[1] + SCALE * 0.06)

    return HandObservation(
        landmarks=tuple(landmarks),
        handedness="Right",
        timestamp_ms=timestamp_ms,
    )


def tip_anchor() -> Landmark:
    """Default index tip of an extended synthetic hand."""
    mcp = (WRIST_POS[0] - 0.35 * SCALE, WRIST_POS[1] - 0.9 * SCALE)
    return (mcp[0], mcp[1] - 1.35 * SCALE)
