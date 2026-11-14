"""JSON message schemas for the edge<->brain WebSocket protocol.

Binary messages on the same socket are JPEG frames and carry no envelope.
"""

from __future__ import annotations

import json
import typing

from pydantic import BaseModel, Field

MESSAGE_TYPE_FIELD = "type"


class HelloMsg(BaseModel):
    """Edge -> Brain: sent once after the socket opens."""

    type: typing.Literal["hello"] = "hello"
    client: str = "edge"
    capture_fps: int = Field(default=30, ge=1, le=120)
    jpeg_quality: int = Field(default=80, ge=1, le=100)


class StatusMsg(BaseModel):
    """Brain -> Edge: periodic pipeline telemetry."""

    type: typing.Literal["status"] = "status"
    fps_inference: float = 0.0
    hands_seen: int = 0
    paused: bool = False
    uptime_s: float = 0.0


class GestureEventMsg(BaseModel):
    """Brain -> Edge: a gesture was recognised (before action mapping)."""

    type: typing.Literal["gesture_event"] = "gesture_event"
    pose: str
    direction: str = "none"


class CommandMsg(BaseModel):
    """Brain -> Edge: execute a whitelisted action on the host."""

    type: typing.Literal["command"] = "command"
    action: str
    keys: list[str] = Field(default_factory=list)
    hold_ms: int = Field(default=0, ge=0)
    origin: str = ""


class LandmarksMsg(BaseModel):
    """Edge <-> Brain: per-frame hand perception.

    Edge -> Brain: detected hand (points normalized, empty = no hand).
    Brain -> Edge: echo with FSM state + cursor coords for the live UI.
    """

    type: typing.Literal["landmarks"] = "landmarks"
    timestamp_ms: int = Field(default=0, ge=0)
    points: list[list[float]] = Field(default_factory=list)  # [[x, y], ...] normalized
    handedness: str = ""
    pose: str = "unknown"
    fsm_state: str = "idle"
    cursor_active: bool = False
    cursor_x: float = 0.0
    cursor_y: float = 0.0


class ActionsMsg(BaseModel):
    """Brain -> Edge: the full action whitelist (registry + custom)."""

    type: typing.Literal["actions"] = "actions"
    actions: list[dict] = Field(default_factory=list)  # ActionSpec dicts


class ReloadConfigMsg(BaseModel):
    """Edge -> Brain: re-read gestures.yaml (mapping changed in the UI)."""

    type: typing.Literal["reload_config"] = "reload_config"


class ErrorMessage(BaseModel):
    """Brain -> Edge: protocol-level failure notice."""

    type: typing.Literal["error"] = "error"
    detail: str = ""


Message = (
    HelloMsg
    | StatusMsg
    | GestureEventMsg
    | LandmarksMsg
    | ActionsMsg
    | ReloadConfigMsg
    | CommandMsg
    | ErrorMessage
)
_MSG_CLASSES: dict[str, type[BaseModel]] = {
    m.model_fields["type"].default: m  # type: ignore[index]
    for m in typing.get_args(Message)
}


def encode_message(msg: Message) -> str:
    return msg.model_dump_json()


def parse_message(raw: str | bytes) -> Message:
    """Parse an incoming text frame into one of the message models."""
    data = json.loads(raw)
    if not isinstance(data, dict) or MESSAGE_TYPE_FIELD not in data:
        raise ValueError("message must be a JSON object with a 'type' field")
    msg_cls = _MSG_CLASSES.get(data[MESSAGE_TYPE_FIELD])
    if msg_cls is None:
        raise ValueError(f"unknown message type: {data[MESSAGE_TYPE_FIELD]!r}")
    return typing.cast(Message, msg_cls.model_validate(data))
