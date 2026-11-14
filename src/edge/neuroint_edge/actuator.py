"""Host actuators: keyboard taps + mouse cursor control via pynput.

Security invariant: a keyboard command is executed only when every key name
maps to a known pynput key. The edge trusts no command payload beyond its
own compiled key table — a hostile or buggy brain cannot type arbitrary
sequences. Mouse actions are a fixed enum (left/right click, absolute move).
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time

from neuroint_protocol.messages import CommandMsg
from pynput.keyboard import Controller, Key
from pynput.mouse import Button

logger = logging.getLogger(__name__)


def _media_key(base_name: str) -> Key:
    """Resolve a media key across pynput's platform-specific attribute names."""
    for candidate in (base_name, f"media_{base_name}"):
        key = getattr(Key, candidate, None)
        if key is not None:
            return key
    raise RuntimeError(f"pynput has no key for {base_name!r} on this platform")


# Known key names -> pynput keys. Extend deliberately, never dynamically.
# Media keys: pynput names differ per platform (Windows prefixes them with
# media_), resolved by _media_key below.
_KEY_TABLE: dict[str, Key] = {
    "alt": Key.alt,
    "tab": Key.tab,
    "shift": Key.shift,
    "ctrl": Key.ctrl,
    "win": Key.cmd,
    "left": Key.left,
    "right": Key.right,
    "up": Key.up,
    "down": Key.down,
    "enter": Key.enter,
    "esc": Key.esc,
    "volume_up": _media_key("volume_up"),
    "volume_down": _media_key("volume_down"),
    "volume_mute": _media_key("volume_mute"),
}

_KEY_NAME_BY_KEY: dict[Key, str] = {key: name for name, key in _KEY_TABLE.items()}


def normalize_key_name(key: Key) -> str | None:
    """Reverse lookup: pynput key -> whitelisted name (for key recording)."""
    return _KEY_NAME_BY_KEY.get(key)


def is_key_allowed(name: str) -> bool:
    return name in _KEY_TABLE


def known_key_names() -> tuple[str, ...]:
    return tuple(_KEY_TABLE)


# Mouse actions: fixed enum, never extended from config.
_MOUSE_BUTTONS = {
    "left_click": Button.left,
    "right_click": Button.right,
}

# Meta actions are handled by the brain itself; the edge must ignore them.
_META_ACTIONS = frozenset({"pause_system"})

# Small delay between key presses: Windows shortcut handling needs the
# modifier to settle before the trigger key (Alt+Tab switcher etiquette).
_KEY_PRESS_DELAY_S = 0.04


class ScreenMapper:
    """Frame-normalized coords -> screen pixels, with edge padding.

    Uses only the inner (1 - 2*padding) region of the frame so the user can
    reach every screen pixel without pushing the hand to the frame border.
    """

    def __init__(self, padding: float = 0.12, size_getter=None) -> None:
        self._padding = padding
        self._size_getter = size_getter or _screen_size

    def to_screen(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        width, height = self._size_getter()
        usable_x = (x_norm - self._padding) / (1.0 - 2.0 * self._padding)
        usable_y = (y_norm - self._padding) / (1.0 - 2.0 * self._padding)
        px = round(min(max(usable_x, 0.0), 1.0) * width)
        py = round(min(max(usable_y, 0.0), 1.0) * height)
        return px, py


def _screen_size() -> tuple[int, int]:
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    raise RuntimeError("screen size detection is implemented for Windows only")


class KeyActuator:
    def __init__(self, controller: Controller | None = None, mouse=None) -> None:
        self._controller = controller
        self._mouse = mouse  # pynput.mouse.Controller, lazily created

    def _ensure_controller(self) -> Controller:
        # pynput's Controller needs a GUI session; construct lazily so tests
        # and probe runs can inject a fake instead.
        if self._controller is None:
            self._controller = Controller()
        return self._controller

    def _ensure_mouse(self):
        if self._mouse is None:
            from pynput.mouse import Controller as MouseController

            self._mouse = MouseController()
        return self._mouse

    def execute(self, command: CommandMsg) -> None:
        if command.action in _META_ACTIONS:
            return
        if command.action in _MOUSE_BUTTONS:
            self._ensure_mouse().click(_MOUSE_BUTTONS[command.action])
            logger.info("executed mouse %s", command.action)
            return
        if not command.keys:
            logger.warning("command %r has no keys, dropped", command.action)
            return
        keys = self._resolve(command.keys)
        if keys is None:
            logger.error("command %r has unmapped keys %s, dropped", command.action, command.keys)
            return
        self._tap_sequence(keys)
        logger.info("executed %s %s", command.action, command.keys)

    def move_cursor(self, x_norm: float, y_norm: float, mapper: ScreenMapper | None = None) -> None:
        """Absolute mouse move from frame-normalized cursor coords."""
        mapper = mapper or ScreenMapper()
        px, py = mapper.to_screen(x_norm, y_norm)
        self._ensure_mouse().position = (px, py)

    def move_cursor_relative(self, dx_norm: float, dy_norm: float, sensitivity: float) -> None:
        """Relative mouse move: fingertip delta x sensitivity x screen size.

        The hand stays near the frame center and steers; sensitivity turns
        small gestures into long screen travel (no arm gymnastics).
        """
        if sys.platform != "win32":
            raise RuntimeError("relative cursor steering is implemented for Windows only")
        width, height = _screen_size()
        px = int(dx_norm * width * sensitivity)
        py = int(dy_norm * height * sensitivity)
        self._ensure_mouse().move(px, py, absolute=False)

    def _resolve(self, names: list[str]) -> list[Key] | None:
        resolved: list[Key] = []
        for name in names:
            key = _KEY_TABLE.get(name)
            if key is None:
                return None
            resolved.append(key)
        return resolved

    def _tap_sequence(self, keys: list[Key]) -> None:
        keyboard = self._ensure_controller()
        for key in keys:
            keyboard.press(key)
            time.sleep(_KEY_PRESS_DELAY_S)
        for key in reversed(keys):
            keyboard.release(key)
