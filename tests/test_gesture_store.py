"""GestureStore: YAML CRUD round-trips and validation."""

import asyncio

import pytest
from neuroint_edge.gesture_store import GestureStore


@pytest.fixture()
def store(tmp_path):
    path = tmp_path / "gestures.yaml"
    path.write_text(
        "rules:\n"
        "  - pose: index_up\n"
        "    direction: right\n"
        "    action: alt_tab\n"
        "custom_actions: []\n",
        encoding="utf-8",
    )
    return GestureStore(path)


def test_read_all(store):
    raw = asyncio.run(store.read_all())
    assert raw["rules"][0]["action"] == "alt_tab"
    assert raw["custom_actions"] == []


def test_add_and_remove_rule(store):
    asyncio.run(store.add_rule("two_up", "left", False, "desktop_left"))
    raw = asyncio.run(store.read_all())
    assert len(raw["rules"]) == 2

    asyncio.run(store.remove_rule("two_up", "left", "desktop_left"))
    raw = asyncio.run(store.read_all())
    assert len(raw["rules"]) == 1


def test_duplicate_rule_rejected(store):
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(store.add_rule("index_up", "right", False, "alt_tab"))


def test_invalid_pose_rejected(store):
    with pytest.raises(ValueError, match="unknown pose"):
        asyncio.run(store.add_rule("middle_finger", "right", False, "alt_tab"))


def test_hold_rule_roundtrip(store):
    asyncio.run(store.add_rule("open_palm", "", True, "pause_system"))
    raw = asyncio.run(store.read_all())
    hold_rules = [r for r in raw["rules"] if r.get("hold")]
    assert len(hold_rules) == 1
    assert hold_rules[0]["pose"] == "open_palm"


def test_custom_action_ids(store):
    id1 = asyncio.run(store.add_custom_action(["ctrl", "shift", "esc"], description="task manager"))
    id2 = asyncio.run(store.add_custom_action(["win"]))
    assert id1 == "custom_1"
    assert id2 == "custom_2"
    raw = asyncio.run(store.read_all())
    assert raw["custom_actions"][0]["keys"] == ["ctrl", "shift", "esc"]

    asyncio.run(store.delete_custom_action(id1))
    raw = asyncio.run(store.read_all())
    assert all(item["id"] != id1 for item in raw["custom_actions"])
