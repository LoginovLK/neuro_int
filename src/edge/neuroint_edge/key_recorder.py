"""Global keyboard combo recorder for the UI ("record a shortcut" button).

pynput's Listener runs OS-wide: the user presses any combination anywhere
(Alt+Tab included) and the recorder captures it as whitelisted key names.
Recording stops on the first non-modifier key press or on cancel.
"""

from __future__ import annotations

import logging
import threading

from pynput import keyboard

from neuroint_edge.actuator import normalize_key_name

logger = logging.getLogger(__name__)

_MODIFIERS = frozenset({"alt", "shift", "ctrl", "win"})


class ComboRecorder:
    """One-shot global combo capture. Thread-safe; callback runs in the
    listener thread — callers bridge into asyncio themselves if needed."""

    def __init__(self) -> None:
        self._listener: keyboard.Listener | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._listener is not None

    def start(self, on_result) -> None:
        """Start capturing; ``on_result(keys: list[str] | None, error: str | None)``
        is called exactly once (keys=None on cancel/invalid combo)."""
        with self._lock:
            if self._listener is not None:
                raise RuntimeError("recording already in progress")
            held_modifiers: list[str] = []
            done = threading.Event()

            def finish(keys: list[str] | None, error: str | None) -> None:
                with self._lock:
                    self._listener = None
                done.set()
                try:
                    on_result(keys, error)
                finally:
                    listener.stop()

            def on_press(key) -> None:
                name = _key_to_name(key)
                if name is None:
                    finish(None, f"key {getattr(key, 'name', '?')!r} is not allowed")
                    return
                if name in _MODIFIERS:
                    if name not in held_modifiers:
                        held_modifiers.append(name)
                    return
                finish([*held_modifiers, name], None)

            def on_cancel() -> None:
                if not done.is_set():
                    finish(None, "cancelled")

            listener = keyboard.Listener(on_press=on_press)
            self._listener = listener
            listener.start()
            logger.info("combo recording started")

    def cancel(self) -> None:
        with self._lock:
            listener = self._listener
            self._listener = None
        if listener is not None:
            listener.stop()
            logger.info("combo recording cancelled")


def _key_to_name(key) -> str | None:
    """Normalize a pynput key event into a whitelisted key name."""
    if isinstance(key, keyboard.Key):
        return normalize_key_name(key)
    # KeyCode (printable characters): map via its vk/name where possible.
    if isinstance(key, keyboard.KeyCode) and key.char:
        char = key.char.lower()
        if char.isalnum() and len(char) == 1:
            return char
    return None
