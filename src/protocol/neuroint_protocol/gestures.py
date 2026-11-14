"""Gesture domain types shared by brain and edge."""

from __future__ import annotations

import enum


class Pose(enum.StrEnum):
    """Static hand poses the brain recognises from landmarks."""

    UNKNOWN = "unknown"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    INDEX_UP = "index_up"
    TWO_UP = "two_up"
    PINCH = "pinch"  # thumb + index tips close together: cursor mode trigger


class Direction(enum.StrEnum):
    """Swipe direction in screen space (from the observer's point of view)."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class GestureEvent:
    """A completed gesture: a pose held steady, then (optionally) a motion.

    ``pose`` is the arming pose, ``direction`` the swipe direction (``NONE``
    for hold-only gestures such as the kill switch).
    """

    __slots__ = ("pose", "direction")

    def __init__(self, pose: Pose, direction: Direction = Direction.NONE) -> None:
        self.pose = pose
        self.direction = direction

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GestureEvent)
            and self.pose == other.pose
            and self.direction == other.direction
        )

    def __hash__(self) -> int:
        return hash((self.pose, self.direction))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"GestureEvent(pose={self.pose.value}, direction={self.direction.value})"
