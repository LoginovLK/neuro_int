"""FSM behavior on synthetic landmark streams (no camera, no MediaPipe)."""

from neuroint_brain.gestures.fsm import GestureFSM
from neuroint_protocol.gestures import Direction, GestureEvent, Pose

from tests.conftest import REPO_ROOT, feed_frames
from tests.synthetic import SCALE, synthetic_hand, tip_anchor


def _swipe_right_factory(t_ms: int):
    """Index tip glides right as time advances: ~0.14 palm per 40ms."""
    progress = max(t_ms - 300, 0) / 1000.0  # starts after arm window
    dx = progress * 7.0 * SCALE  # reaches ~2.9 palm widths
    tip = (tip_anchor()[0] + dx, tip_anchor()[1])
    return synthetic_hand(Pose.INDEX_UP, t_ms, index_tip_override=tip)


def test_swipe_right_emits_once() -> None:
    fsm = GestureFSM()
    events = feed_frames(fsm, _swipe_right_factory, count=25, dt_ms=40)
    assert GestureEvent(Pose.INDEX_UP, Direction.RIGHT) in events
    assert len(events) == 1  # cooldown suppresses duplicates


def test_casual_pose_pass_does_not_arm() -> None:
    # Pose visible for only 150ms (< arm_hold 220ms): nothing must fire.
    fsm = GestureFSM()
    events = feed_frames(fsm, lambda t: synthetic_hand(Pose.INDEX_UP, t), count=4, dt_ms=50)
    assert events == []


def test_slow_drift_does_not_swipe() -> None:
    # Tip moves far but slowly (2.5 palm/s): the FSM re-anchors each window
    # (600ms -> 1.5 palm max), under the swipe threshold, so nothing fires.
    fsm = GestureFSM()
    step = SCALE * 0.1  # 0.1 palm per frame

    def factory(t_ms: int):
        idx = t_ms // 40
        tip = (tip_anchor()[0] + idx * step, tip_anchor()[1])
        return synthetic_hand(Pose.INDEX_UP, t_ms, index_tip_override=tip)

    events = feed_frames(fsm, factory, count=40, dt_ms=40)
    assert events == []


def test_open_palm_hold_fires_kill_switch() -> None:
    fsm = GestureFSM()
    # 14 * 120ms = 1680ms: arm window + kill_hold (1100ms) + margin.
    events = feed_frames(fsm, lambda t: synthetic_hand(Pose.OPEN_PALM, t), count=14, dt_ms=120)
    assert GestureEvent(Pose.OPEN_PALM, Direction.NONE) in events
    assert events.count(GestureEvent(Pose.OPEN_PALM, Direction.NONE)) == 1


def test_pose_change_resets() -> None:
    fsm = GestureFSM()
    # Arm index_up for 300ms, then break the pose.
    feed_frames(fsm, lambda t: synthetic_hand(Pose.INDEX_UP, t), count=6, dt_ms=50)
    events = feed_frames(fsm, lambda t: synthetic_hand(Pose.TWO_UP, t), count=3, dt_ms=40)
    assert events == []


def test_hand_loss_resets() -> None:
    fsm = GestureFSM()
    feed_frames(fsm, lambda t: synthetic_hand(Pose.INDEX_UP, t), count=6, dt_ms=50)
    # Hand disappears long enough to exceed hand_lost_reset_ms.
    events: list[GestureEvent] = []
    for _ in range(15):
        events.extend(fsm.feed(None))
    assert events == []
    # Losing the hand mid-armed state must fully reset: no stale swipe.
    events = feed_frames(fsm, _swipe_right_factory, count=25, dt_ms=40, start_ms=1500)
    assert len(events) == 1  # fresh recognition still works


def test_config_paths_exist() -> None:
    assert (REPO_ROOT / "configs" / "gestures.yaml").exists()
