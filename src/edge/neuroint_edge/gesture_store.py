"""gestures.yaml persistence: read, add, remove rules and custom actions.

The file is the single source of truth for the mapping; the brain re-reads
it on a reload_config message, so UI edits take effect without a rebuild.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib

import yaml
from neuroint_protocol.gestures import Direction, Pose

logger = logging.getLogger(__name__)

_FILE_HEADER = """# Gesture -> action mapping. Managed by the neuro_int UI.
#
# Manual edits are fine too: apply them with `make restart-config` or from
# the UI. Format:
#   rules: [{pose, direction|hold, action}, ...]
#   custom_actions: [{id, keys, description}, ...]
"""


class GestureStore:
    """Blocking YAML read/write; every call is guarded by an asyncio lock."""

    def __init__(self, path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._lock = asyncio.Lock()

    def _load(self) -> dict:
        raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        raw.setdefault("rules", [])
        raw.setdefault("custom_actions", [])
        return raw

    def _save(self, raw: dict) -> None:
        text = _FILE_HEADER + yaml.safe_dump(
            {"rules": raw["rules"], "custom_actions": raw["custom_actions"]},
            sort_keys=False,
            allow_unicode=True,
        )
        self._path.write_text(text, encoding="utf-8")

    async def read_all(self) -> dict:
        async with self._lock:
            return self._load()

    async def next_custom_id(self) -> str:
        async with self._lock:
            raw = self._load()
            existing = {item.get("id", "") for item in raw["custom_actions"]}
            for n in range(1, 1000):
                candidate = f"custom_{n}"
                if candidate not in existing:
                    return candidate
            raise RuntimeError("too many custom actions")

    async def add_rule(self, pose: str, direction: str, hold: bool, action: str) -> None:
        """Append a mapping rule. Raises ValueError on invalid input."""
        if pose not in {p.value for p in Pose}:
            raise ValueError(f"unknown pose {pose!r}")
        if hold:
            motion = {"hold": True}
        elif direction in {d.value for d in Direction} and direction != Direction.NONE.value:
            motion = {"direction": direction}
        else:
            raise ValueError(f"invalid motion {direction!r}")
        if not action:
            raise ValueError("empty action")
        async with self._lock:
            raw = self._load()
            for item in raw["rules"]:
                if (
                    item.get("pose") == pose
                    and item.get("action") == action
                    and item.get("direction") == motion.get("direction")
                    and bool(item.get("hold")) == hold
                ):
                    raise ValueError("identical rule already exists")
            raw["rules"].append({"pose": pose, **motion, "action": action})
            self._save(raw)
        logger.info("rule added: %s/%s -> %s", pose, "hold" if hold else direction, action)

    async def remove_rule(self, pose: str, motion: str, action: str) -> None:
        """Remove the rule matching pose+motion('left'|'hold'|...)+action."""
        async with self._lock:
            raw = self._load()
            before = len(raw["rules"])
            raw["rules"] = [
                item
                for item in raw["rules"]
                if not (
                    item.get("pose") == pose
                    and item.get("action") == action
                    and (item.get("direction") == motion or (motion == "hold" and item.get("hold")))
                )
            ]
            if len(raw["rules"]) == before:
                raise ValueError("rule not found")
            self._save(raw)
        logger.info("rule removed: %s/%s -> %s", pose, motion, action)

    async def add_custom_action(self, keys: list[str], description: str = "") -> str:
        """Register a UI-recorded keyboard combo; returns its generated id."""
        if not keys:
            raise ValueError("empty key combo")
        async with self._lock:
            raw = self._load()
            existing = {item.get("id", "") for item in raw["custom_actions"]}
            action_id = None
            for n in range(1, 1000):
                candidate = f"custom_{n}"
                if candidate not in existing:
                    action_id = candidate
                    break
            if action_id is None:
                raise RuntimeError("too many custom actions")
            raw["custom_actions"].append(
                {"id": action_id, "keys": keys, "description": description}
            )
            self._save(raw)
        logger.info("custom action %s = %s", action_id, keys)
        return action_id

    async def delete_custom_action(self, action_id: str) -> None:
        """Remove a custom action and any rules that reference it."""
        async with self._lock:
            raw = self._load()
            before = len(raw["custom_actions"])
            raw["custom_actions"] = [
                item for item in raw["custom_actions"] if item.get("id") != action_id
            ]
            if len(raw["custom_actions"]) == before:
                raise ValueError("custom action not found")
            raw["rules"] = [item for item in raw["rules"] if item.get("action") != action_id]
            self._save(raw)
        logger.info("custom action %s removed (with its rules)", action_id)
