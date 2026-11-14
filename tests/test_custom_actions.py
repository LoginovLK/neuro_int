"""Custom actions: yaml registration + mapping through the whitelist."""

from neuroint_brain.actions.mapper import load_mapping
from neuroint_protocol.gestures import Direction, GestureEvent, Pose
from neuroint_protocol.messages import CommandMsg


def _write_config(tmp_path) -> object:
    path = tmp_path / "gestures.yaml"
    path.write_text(
        "custom_actions:\n"
        "  - id: custom_1\n"
        "    keys: [ctrl, shift, esc]\n"
        "    description: task manager\n"
        "rules:\n"
        "  - pose: fist\n"
        "    direction: right\n"
        "    action: custom_1\n",
        encoding="utf-8",
    )
    return path


def test_custom_action_maps_with_keys(tmp_path):
    config = _write_config(tmp_path)
    mapper = load_mapping(config)
    cmd = mapper.map(GestureEvent(Pose.FIST, Direction.RIGHT))
    assert isinstance(cmd, CommandMsg)
    assert cmd.action == "custom_1"
    assert cmd.keys == ["ctrl", "shift", "esc"]


def test_custom_action_in_whitelist(tmp_path):
    config = _write_config(tmp_path)
    load_mapping(config)
    from neuroint_brain.actions.registry import get_action

    spec = get_action("custom_1")
    assert spec is not None
    assert spec.kind == "keys"


def test_reload_clears_stale_customs(tmp_path):
    config = _write_config(tmp_path)
    load_mapping(config)
    # Rewrite without custom_actions: old ids must disappear from the registry.
    config.write_text(
        "custom_actions: []\nrules:\n  - pose: fist\n    direction: right\n    action: alt_tab\n",
        encoding="utf-8",
    )
    load_mapping(config)
    from neuroint_brain.actions.registry import get_action

    assert get_action("custom_1") is None
