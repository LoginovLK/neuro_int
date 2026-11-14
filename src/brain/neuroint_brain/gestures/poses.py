"""Static pose classification from raw landmarks.

Thumb extension is deliberately ignored for fist/palm poses — at PS3 Eye
resolution it is unreliable — but thumb tip proximity to the index tip
defines the PINCH pose (cursor mode trigger).
"""

from __future__ import annotations

from neuroint_protocol.gestures import Pose

from neuroint_brain.vision.hands import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    HandObservation,
)

# Thumb-index distance threshold, in palm widths, for the pinch pose.
PINCH_RATIO = 0.45


def is_pinch(hand: HandObservation) -> bool:
    """Thumb and index tips close together (cursor grab gesture)."""
    palm = hand.palm_size()
    if palm < 1e-4:
        return False
    thumb = hand.landmarks[THUMB_TIP]
    index = hand.landmarks[INDEX_TIP]
    return ((thumb[0] - index[0]) ** 2 + (thumb[1] - index[1]) ** 2) ** 0.5 < palm * PINCH_RATIO


def classify_pose(hand: HandObservation) -> Pose:
    """Map a hand observation to one of the supported static poses."""
    if is_pinch(hand):
        return Pose.PINCH

    extended = [
        hand.finger_extended(INDEX_TIP, INDEX_PIP),
        hand.finger_extended(MIDDLE_TIP, MIDDLE_PIP),
        hand.finger_extended(RING_TIP, RING_PIP),
        hand.finger_extended(PINKY_TIP, PINKY_PIP),
    ]

    if all(extended):
        return Pose.OPEN_PALM
    if not any(extended):
        return Pose.FIST
    if extended == [True, False, False, False]:
        return Pose.INDEX_UP
    if extended == [True, True, False, False]:
        return Pose.TWO_UP
    return Pose.UNKNOWN
