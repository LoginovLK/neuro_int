"""Gesture -> action mapping loaded from configs/gestures.yaml.

On every load the ``custom_actions`` section is registered into the action
registry (UI-defined keyboard combos) and old custom entries are dropped,
so a config reload fully rebuilds the effective whitelist.
"""

from __future__ import annotations

import logging
import pathlib

import yaml
from neuroint_protocol.gestures import Direction, GestureEvent, Pose
from neuroint_protocol.messages import CommandMsg

from neuroint_brain.actions.registry import clear_custom, get_action, register_custom

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {d.value for d in Direction}
_VALID_POSES = {p.value for p in Pose}


class _Rule:
    __slots__ = ("pose", "direction", "hold", "action")

    def __init__(self, pose: Pose, direction: Direction, hold: bool, action: str) -> None:
        self.pose = pose
        self.direction = direction
        self.hold = hold
        self.action = action

    def matches(self, event: GestureEvent) -> bool:
        if self.pose != event.pose:
            return False
        if self.hold:
            return event.direction == Direction.NONE
        return event.direction == self.direction


class ActionMapper:
    """Applies mapping rules; unknown events are logged and dropped."""

    def __init__(self, rules: list[_Rule]) -> None:
        self._rules = rules

    def map(self, event: GestureEvent) -> CommandMsg | None:
        for rule in self._rules:
            if rule.matches(event):
                spec = get_action(rule.action)
                if spec is None:
                    logger.error("mapping refers to unknown action %r", rule.action)
                    continue
                if spec.kind == "meta":
                    return CommandMsg(action=spec.action, origin=_origin(event))
                if spec.kind == "mouse":
                    return CommandMsg(action=spec.action, origin=_origin(event))
                return CommandMsg(
                    action=spec.action,
                    keys=list(spec.keys),
                    origin=_origin(event),
                )
        logger.info("no mapping for gesture %s/%s", event.pose.value, event.direction.value)
        return None


def load_mapping(path: str | pathlib.Path) -> ActionMapper:
    """Parse and validate gestures.yaml (fail fast on bad rules)."""
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    _register_custom_actions(raw)
    rules = _parse_rules(raw)
    logger.info("loaded %d gesture mapping rules from %s", len(rules), path)
    return ActionMapper(rules)


def _register_custom_actions(raw: dict) -> None:
    clear_custom()
    for i, item in enumerate(raw.get("custom_actions", [])):
        action_id = item.get("id")
        keys = item.get("keys", [])
        if not action_id or not keys:
            raise ValueError(f"gestures.yaml custom_actions #{i}: needs id and keys")
        register_custom(action_id, tuple(keys), description=item.get("description", ""))
    if raw.get("custom_actions"):
        logger.info("registered %d custom actions", len(raw["custom_actions"]))


def _parse_rules(raw: dict) -> list[_Rule]:
    rules: list[_Rule] = []
    for i, item in enumerate(raw.get("rules", [])):
        pose_value = item.get("pose")
        if pose_value not in _VALID_POSES:
            raise ValueError(f"gestures.yaml rule #{i}: unknown pose {pose_value!r}")
        action = item.get("action")
        if get_action(action) is None:
            raise ValueError(f"gestures.yaml rule #{i}: unknown action {action!r}")
        if item.get("hold"):
            rule = _Rule(Pose(pose_value), Direction.NONE, True, action)
        else:
            direction = item.get("direction")
            if direction not in _VALID_DIRECTIONS or direction == Direction.NONE.value:
                raise ValueError(
                    f"gestures.yaml rule #{i}: swipe rule needs direction in "
                    f"{sorted(_VALID_DIRECTIONS - {Direction.NONE.value})}"
                )
            rule = _Rule(Pose(pose_value), Direction(direction), False, action)
        rules.append(rule)
    return rules


def _origin(event: GestureEvent) -> str:
    return f"{event.pose.value}:{event.direction.value}"
