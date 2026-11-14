"""Shared wire contract between the brain (container) and edge (host agent).

Text WebSocket messages are JSON payloads from :mod:`neuroint_protocol.messages`;
binary WebSocket messages are JPEG-encoded camera frames.
"""

from neuroint_protocol.gestures import Direction, GestureEvent, Pose
from neuroint_protocol.messages import (
    ActionsMsg,
    CommandMsg,
    ErrorMessage,
    GestureEventMsg,
    HelloMsg,
    LandmarksMsg,
    ReloadConfigMsg,
    StatusMsg,
    parse_message,
)

__all__ = [
    "ActionsMsg",
    "CommandMsg",
    "Direction",
    "ErrorMessage",
    "GestureEvent",
    "GestureEventMsg",
    "HelloMsg",
    "LandmarksMsg",
    "Pose",
    "ReloadConfigMsg",
    "StatusMsg",
    "parse_message",
]
