"""Actuator: whitelist re-validation and tap ordering (fake controller)."""

from neuroint_edge.actuator import KeyActuator
from neuroint_protocol.messages import CommandMsg


class FakeController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def press(self, key) -> None:
        self.calls.append(("press", str(key)))

    def release(self, key) -> None:
        self.calls.append(("release", str(key)))


def _actuator() -> tuple[KeyActuator, FakeController]:
    controller = FakeController()
    return KeyActuator(controller=controller), controller


def test_alt_tab_sequence() -> None:
    actuator, controller = _actuator()
    actuator.execute(CommandMsg(action="alt_tab", keys=["alt", "tab"], origin="test"))
    kinds = [k for k, _ in controller.calls]
    assert kinds == ["press", "press", "release", "release"]


def test_unknown_key_dropped_without_execution() -> None:
    actuator, controller = _actuator()
    actuator.execute(CommandMsg(action="alt_tab", keys=["alt", "f13"], origin="test"))
    assert controller.calls == []


def test_no_keys_dropped() -> None:
    actuator, controller = _actuator()
    actuator.execute(CommandMsg(action="alt_tab", keys=[], origin="test"))
    assert controller.calls == []


def test_meta_action_ignored() -> None:
    actuator, controller = _actuator()
    actuator.execute(CommandMsg(action="pause_system", keys=["alt"], origin="test"))
    assert controller.calls == []


def test_desktop_shortcut_order() -> None:
    actuator, controller = _actuator()
    actuator.execute(
        CommandMsg(action="desktop_right", keys=["ctrl", "win", "right"], origin="test")
    )
    kinds = [k for k, _ in controller.calls]
    assert kinds == ["press", "press", "press", "release", "release", "release"]
