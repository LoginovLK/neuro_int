"""Mapping rules: real gestures.yaml + validation failures."""

import pytest
from neuroint_brain.actions.mapper import load_mapping
from neuroint_protocol.gestures import Direction, GestureEvent, Pose

from tests.conftest import REPO_ROOT

REAL_CONFIG = REPO_ROOT / "configs" / "gestures.yaml"


@pytest.fixture()
def mapper():
    return load_mapping(REAL_CONFIG)


def test_real_config_loads(mapper) -> None:
    assert len(mapper._rules) >= 5


def test_index_right_is_alt_tab(mapper) -> None:
    cmd = mapper.map(GestureEvent(Pose.INDEX_UP, Direction.RIGHT))
    assert cmd is not None
    assert cmd.action == "alt_tab"
    assert cmd.keys == ["alt", "tab"]
    assert cmd.origin == "index_up:right"


def test_index_left_is_alt_shift_tab(mapper) -> None:
    cmd = mapper.map(GestureEvent(Pose.INDEX_UP, Direction.LEFT))
    assert cmd is not None
    assert cmd.action == "alt_shift_tab"


def test_two_up_desktops(mapper) -> None:
    cmd_right = mapper.map(GestureEvent(Pose.TWO_UP, Direction.RIGHT))
    cmd_left = mapper.map(GestureEvent(Pose.TWO_UP, Direction.LEFT))
    assert cmd_right is not None and cmd_right.action == "desktop_right"
    assert cmd_left is not None and cmd_left.action == "desktop_left"


def test_open_palm_hold_is_pause(mapper) -> None:
    cmd = mapper.map(GestureEvent(Pose.OPEN_PALM, Direction.NONE))
    assert cmd is not None
    assert cmd.action == "pause_system"
    assert cmd.keys == []  # meta action, never typed on the host


def test_unmapped_event_returns_none(mapper) -> None:
    assert mapper.map(GestureEvent(Pose.FIST, Direction.RIGHT)) is None
    assert mapper.map(GestureEvent(Pose.INDEX_UP, Direction.UP)) is None


def test_unknown_action_rejected(tmp_path) -> None:
    bad = tmp_path / "gestures.yaml"
    bad.write_text(
        "rules:\n  - pose: index_up\n    direction: right\n    action: rm_rf_everything\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown action"):
        load_mapping(bad)


def test_swipe_rule_without_direction_rejected(tmp_path) -> None:
    bad = tmp_path / "gestures.yaml"
    bad.write_text(
        "rules:\n  - pose: index_up\n    action: alt_tab\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="direction"):
        load_mapping(bad)
