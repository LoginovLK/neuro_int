"""Whitelisted host actions.

Every command sent to the edge agent must resolve to an entry here. Keys use
pynput names; the edge validates them again on its side (defense in depth).

Kinds:
  keys  - a keyboard shortcut sequence (edge taps the keys)
  mouse - a mouse primitive handled by the edge actuator (no keys)
  meta  - handled by the brain itself, never reaches the host

Custom actions come from the ``custom_actions`` section of gestures.yaml
(added through the UI) and are registered dynamically on every config load.
"""

from __future__ import annotations

import dataclasses
import threading


@dataclasses.dataclass(frozen=True)
class ActionSpec:
    action: str
    keys: tuple[str, ...] = ()
    kind: str = "keys"  # keys | mouse | meta
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "keys": list(self.keys),
            "kind": self.kind,
            "description": self.description,
        }


_BUILTIN_CATALOG: dict[str, ActionSpec] = {
    spec.action: spec
    for spec in (
        ActionSpec(
            "alt_tab",
            keys=("alt", "tab"),
            description="Switch to the next window",
        ),
        ActionSpec(
            "alt_shift_tab",
            keys=("alt", "shift", "tab"),
            description="Switch to the previous window",
        ),
        ActionSpec(
            "desktop_left",
            keys=("ctrl", "win", "left"),
            description="Previous virtual desktop",
        ),
        ActionSpec(
            "desktop_right",
            keys=("ctrl", "win", "right"),
            description="Next virtual desktop",
        ),
        ActionSpec("volume_up", keys=("volume_up",), description="Volume up"),
        ActionSpec("volume_down", keys=("volume_down",), description="Volume down"),
        ActionSpec("toggle_mute", keys=("volume_mute",), description="Mute toggle"),
        ActionSpec(
            "pause_system",
            kind="meta",
            description="Toggle gesture control on/off (kill switch)",
        ),
        ActionSpec(
            "left_click",
            kind="mouse",
            description="Mouse left click (cursor mode)",
        ),
        ActionSpec(
            "right_click",
            kind="mouse",
            description="Mouse right click (cursor mode)",
        ),
    )
}

_CUSTOM_CATALOG: dict[str, ActionSpec] = {}
_CUSTOM_LOCK = threading.Lock()


def get_action(name: str) -> ActionSpec | None:
    with _CUSTOM_LOCK:
        if name in _CUSTOM_CATALOG:
            return _CUSTOM_CATALOG[name]
    return _BUILTIN_CATALOG.get(name)


def list_actions() -> tuple[ActionSpec, ...]:
    with _CUSTOM_LOCK:
        return tuple(_BUILTIN_CATALOG.values()) + tuple(_CUSTOM_CATALOG.values())


def register_custom(action_id: str, keys: tuple[str, ...], description: str = "") -> ActionSpec:
    """Register a UI-defined keyboard action (validated key names required)."""
    spec = ActionSpec(action=action_id, keys=keys, kind="keys", description=description)
    with _CUSTOM_LOCK:
        _CUSTOM_CATALOG[action_id] = spec
    return spec


def clear_custom() -> None:
    """Drop all custom actions (called before each gestures.yaml reload)."""
    with _CUSTOM_LOCK:
        _CUSTOM_CATALOG.clear()
