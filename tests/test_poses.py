from neuroint_brain.gestures.poses import classify_pose
from neuroint_protocol.gestures import Pose

from tests.synthetic import synthetic_hand


def test_fist() -> None:
    assert classify_pose(synthetic_hand(Pose.FIST, 0)) == Pose.FIST


def test_open_palm() -> None:
    assert classify_pose(synthetic_hand(Pose.OPEN_PALM, 0)) == Pose.OPEN_PALM


def test_index_up() -> None:
    assert classify_pose(synthetic_hand(Pose.INDEX_UP, 0)) == Pose.INDEX_UP


def test_two_up() -> None:
    assert classify_pose(synthetic_hand(Pose.TWO_UP, 0)) == Pose.TWO_UP


def test_index_tip_override_keeps_pose() -> None:
    # Moving the index tip far away (swipe) must not change classification.
    hand = synthetic_hand(Pose.INDEX_UP, 0, index_tip_override=(0.9, 0.3))
    assert classify_pose(hand) == Pose.INDEX_UP
