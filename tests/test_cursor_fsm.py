"""Cursor mode FSM behavior: pinch steers, quick regrip clicks."""

import time

from neuroint_brain.gestures.fsm import GestureFSM
from neuroint_protocol.gestures import Direction, GestureEvent, Pose

from tests.synthetic import synthetic_hand


def test_pinch_classifies() -> None:
    hand = synthetic_hand(Pose.PINCH, 0, pinch=True)
    assert hand.landmarks  # sanity


def test_pinch_enters_cursor_mode() -> None:
    fsm = GestureFSM()
    for t in (0, 33, 66):
        events = fsm.feed(synthetic_hand(Pose.PINCH, t, pinch=True))
        assert events == []
    assert fsm.state_name == "cursor"
    assert fsm.cursor_active is True
    assert 0.0 < fsm.cursor_x < 1.0


def test_cursor_coordinates_move() -> None:
    fsm = GestureFSM()
    fsm.feed(synthetic_hand(Pose.PINCH, 0, pinch=True))
    fsm.feed(synthetic_hand(Pose.PINCH, 33, pinch=True))
    x_before = fsm.cursor_x
    # Move the whole steering point right by one palm.
    hand = synthetic_hand(Pose.PINCH, 66, pinch=True, index_tip_override=(0.7, 0.3))
    fsm.feed(hand)
    assert fsm.cursor_x > x_before


def test_quick_regrip_emits_click() -> None:
    fsm = GestureFSM()
    fsm.feed(synthetic_hand(Pose.PINCH, 0, pinch=True))
    fsm.feed(synthetic_hand(Pose.PINCH, 33, pinch=True))
    # Release (different pose) briefly...
    fsm.feed(synthetic_hand(Pose.INDEX_UP, 66))
    # ...and re-grip fast: must emit exactly one click (PINCH/NONE).
    events = fsm.feed(synthetic_hand(Pose.PINCH, 100, pinch=True))
    assert GestureEvent(Pose.PINCH, Direction.NONE) in events
    assert fsm.state_name == "cursor"


def test_regrip_after_window_does_not_click() -> None:
    fsm = GestureFSM()
    fsm.feed(synthetic_hand(Pose.PINCH, 0, pinch=True))
    fsm.feed(synthetic_hand(Pose.INDEX_UP, 33))
    # Gap beyond pending_click (380ms) but inside cursor_lost (600ms):
    # cursor mode survives, but no click — not an intentional regrip.
    events = fsm.feed(synthetic_hand(Pose.INDEX_UP, 500))
    assert events == []
    events = fsm.feed(synthetic_hand(Pose.PINCH, 520, pinch=True))
    assert events == []
    assert fsm.state_name == "cursor"


def test_cursor_blocks_swipes_and_resets() -> None:
    fsm = GestureFSM()
    # Arm a swipe gesture first, then pinch into cursor mode.
    for t in range(0, 300, 50):
        fsm.feed(synthetic_hand(Pose.INDEX_UP, t))
    events = fsm.feed(synthetic_hand(Pose.PINCH, 300, pinch=True))
    assert events == []  # armed swipe dropped, cursor takes over
    assert fsm.state_name == "cursor"
    # Swipe attempt inside cursor mode must not fire.
    for t in range(330, 700, 40):
        events = fsm.feed(synthetic_hand(Pose.INDEX_UP, t, index_tip_override=(0.9, 0.3)))
    assert events == []


def test_cursor_exits_to_idle() -> None:
    fsm = GestureFSM()
    fsm.feed(synthetic_hand(Pose.PINCH, 0, pinch=True))
    fsm.feed(synthetic_hand(Pose.INDEX_UP, 33))
    fsm.feed(synthetic_hand(Pose.INDEX_UP, 500))  # gap 467ms: still tolerated
    assert fsm.state_name != "idle"
    fsm.feed(synthetic_hand(Pose.INDEX_UP, 800))  # gap 767ms > cursor_lost_ms
    assert fsm.state_name == "idle"
    assert fsm.cursor_active is False


def test_hand_gone_exits_cursor() -> None:
    fsm = GestureFSM()
    fsm.feed(synthetic_hand(Pose.PINCH, 0, pinch=True))
    fsm.feed(None)  # CURSOR -> PENDING: the lost-hand clock starts here
    time.sleep(0.62)
    fsm.feed(None)  # gap exceeded cursor_lost_ms -> exit
    assert fsm.state_name == "idle"
    assert fsm.cursor_active is False


def test_blur_blink_keeps_cursor_mode() -> None:
    # A few degraded frames (fast hand movement) must NOT exit cursor mode:
    # this is exactly the "bones disappear while steering" complaint.
    fsm = GestureFSM()
    fsm.feed(synthetic_hand(Pose.PINCH, 0, pinch=True))
    fsm.feed(synthetic_hand(Pose.UNKNOWN, 33))  # blurred / unrecognizable
    fsm.feed(synthetic_hand(Pose.UNKNOWN, 66))
    assert fsm.state_name != "idle"
    events = fsm.feed(synthetic_hand(Pose.PINCH, 100, pinch=True))
    assert events == []  # resumed steering, no click (blur gap > pending_click)
    assert fsm.state_name == "cursor"
    assert fsm.cursor_active is True


def test_hold_fires_once_per_physical_hold() -> None:
    # Kill switch must NOT toggle repeatedly while the palm stays open:
    # it fires once, locks until the pose changes, then re-arms.
    fsm = GestureFSM()
    t = 0
    first: list = []
    for _ in range(14):  # 14 * 120ms = 1.68s of an open palm
        first.extend(fsm.feed(synthetic_hand(Pose.OPEN_PALM, t)))
        t += 120
    assert first == [GestureEvent(Pose.OPEN_PALM, Direction.NONE)]
    assert fsm.state_name == "hold_lock"

    # Palm still open for a long time: no second event.
    again: list = []
    for _ in range(10):
        again.extend(fsm.feed(synthetic_hand(Pose.OPEN_PALM, t)))
        t += 120
    assert again == []
    assert fsm.state_name == "hold_lock"

    # Pose changes -> re-arm; the next hold fires again (pause toggle works).
    fsm.feed(synthetic_hand(Pose.FIST, t))
    t += 40
    second: list = []
    for _ in range(14):
        second.extend(fsm.feed(synthetic_hand(Pose.OPEN_PALM, t)))
        t += 120
    assert second == [GestureEvent(Pose.OPEN_PALM, Direction.NONE)]
