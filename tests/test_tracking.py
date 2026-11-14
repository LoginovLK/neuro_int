"""Tracked-ROI construction from previous landmarks."""

from math import atan2, cos, degrees, hypot, radians, sin

from neuroint_cursor.tracking import rect_from_landmarks
from test_pointing import _rotate

# Square frame: the synthetic hand is built in a unit square, a square
# frame keeps norm->pixel scaling uniform so angles pass through exactly.
FRAME_W = FRAME_H = 1000


def _vertical_hand():
    """21-point hand with the middle-finger MCP exactly above the wrist."""
    pts = [(0.50, 0.80)]  # wrist (0)
    pts += [(0.42, 0.76), (0.40, 0.72), (0.38, 0.70), (0.37, 0.68)]  # thumb
    for fx in (0.46, 0.50, 0.54, 0.58):  # index, middle, ring, pinky
        pts += [(fx, 0.66), (fx + 0.002, 0.54), (fx + 0.001, 0.44), (fx, 0.36)]
    assert len(pts) == 21
    return pts


def test_pointing_up_gives_zero_rotation():
    pts = _vertical_hand()
    rect, d = rect_from_landmarks(pts, FRAME_W, FRAME_H, None, None)
    assert rect is not None
    cx, cy, w, h, deg = rect
    assert abs(deg) < 1.0  # pointing up -> no crop rotation
    assert w == h
    xs = [x * FRAME_W for x, _ in pts]
    ys = [y * FRAME_H for _, y in pts]
    diag = hypot(max(xs) - min(xs), max(ys) - min(ys))
    assert w >= diag  # ROI covers the whole hand with margin


def test_rotation_follows_hand_angle():
    for deg in (30, -45):
        pts = _rotate(_vertical_hand(), deg)
        rect, _ = rect_from_landmarks(pts, FRAME_W, FRAME_H, None, None)
        assert rect is not None
        assert abs(rect[4] - deg) < 2.0, f"expected ~{deg}, got {rect[4]}"


def test_direction_smoothing_is_stable():
    pts = _vertical_hand()
    _, d1 = rect_from_landmarks(pts, FRAME_W, FRAME_H, None, None)
    _, d2 = rect_from_landmarks(pts, FRAME_W, FRAME_H, d1, None)
    assert abs(d1[0] - d2[0]) < 0.1 and abs(d1[1] - d2[1]) < 0.1
    assert abs(hypot(d2[0], d2[1]) - 1.0) < 1e-6  # stays a unit vector


def test_wrist_centered_roi():
    pts = _vertical_hand()
    rect, _ = rect_from_landmarks(pts, FRAME_W, FRAME_H, None, None)
    cx, cy = rect[0], rect[1]
    # bbox center of an up-pointing hand sits above the wrist (y=0.8*1000)
    assert cy < 0.8 * FRAME_H
    assert 0 < cx < FRAME_W


def test_growth_limited_by_prev_rect():
    """A garbage landmark frame must not blow the crop up."""
    pts = _vertical_hand()
    prev = [500, 600, 200, 200, 0.0]  # small previous ROI
    rect, _ = rect_from_landmarks(pts, FRAME_W, FRAME_H, None, prev)
    assert rect is not None
    # raw size would be ~1.8*diag (~600px); limiter caps growth to 1.25x
    assert rect[2] <= 1.25 * 200 + 1


def test_center_jump_limited():
    pts = _vertical_hand()  # bbox center ~ (475, 580)
    prev = [900, 900, 200, 200, 0.0]  # far away
    rect, _ = rect_from_landmarks(pts, FRAME_W, FRAME_H, None, prev)
    assert rect is not None
    dx = rect[0] - prev[0]
    dy = rect[1] - prev[1]
    dist = hypot(dx, dy)
    assert dist <= 0.6 * 200 + 1  # center pulled to the jump limit


def test_rotation_roundtrip():
    deg = 30
    a = radians(deg)
    assert abs(degrees(atan2(sin(a), cos(a))) - deg) < 1e-9
