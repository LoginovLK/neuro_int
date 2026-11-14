"""MediaPipe-faithful rect-from-landmarks: geometry properties.

Source of truth (ported verbatim into neuroint_cursor.tracking):
mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc
and hand_landmark_landmarks_to_roi.pbtxt (scale 2.0, shift_y -0.1, square_long).
"""

from math import cos, isclose, radians, sin

from neuroint_cursor.tracking import TRACK_SCALE, normalize_radians, rect_from_landmarks
from test_pointing import _rotate

# Square frame: the synthetic hand is built in a unit square, a square
# frame keeps norm->pixel scaling uniform so angles pass through exactly.
FRAME = (1000, 1000)
PARTIAL = (0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)


def _finger(wx, wy, dx, dy, lengths):
    """4 landmarks along direction (dx, dy) at per-joint distances."""
    return [(wx + dx * ln, wy + dy * ln) for ln in lengths]


def _upright_hand(wrist=(0.5, 0.8)):
    """Synthetic hand: wrist at the bottom, fingers straight up (y down)."""
    wx, wy = wrist
    pts = [(wx, wy)]  # 0 wrist
    pts += _finger(wx, wy, -0.3, -0.2, (0.10, 0.17, 0.25, 0.33))  # thumb 1-4
    pts += _finger(wx, wy, -0.02, -0.10, (0.10, 0.16, 0.21, 0.26))  # index
    pts += _finger(wx, wy, 0.0, -0.11, (0.11, 0.17, 0.22, 0.27))  # middle
    pts += _finger(wx, wy, 0.02, -0.10, (0.10, 0.16, 0.21, 0.25))  # ring
    pts += _finger(wx, wy, 0.04, -0.09, (0.09, 0.14, 0.18, 0.22))  # pinky
    assert len(pts) == 21
    return pts


def test_normalize_radians_wraps():
    from math import pi, tau

    assert isclose(normalize_radians(3.5 * pi), 3.5 * pi - 2 * tau, abs_tol=1e-6)
    assert isclose(normalize_radians(-0.1), -0.1, abs_tol=1e-9)
    assert isclose(normalize_radians(0.5 * pi), 0.5 * pi, abs_tol=1e-9)


def test_upright_hand_zero_rotation():
    rect = rect_from_landmarks(_upright_hand(), *FRAME)
    assert rect is not None
    # wrist -> mean PIP is straight up: rotation must be 0.
    assert abs(rect[4]) < 1e-6


def test_tilted_hand_rotation_follows():
    rect = rect_from_landmarks(_rotate(_upright_hand(), 30.0), *FRAME)
    assert rect is not None
    assert isclose(rect[4] % 360.0, 30.0, abs_tol=2.0), rect[4]


def test_rect_follows_hand():
    base = rect_from_landmarks(_upright_hand(), *FRAME)
    moved = rect_from_landmarks(_upright_hand(wrist=(0.7, 0.8)), *FRAME)
    assert base is not None and moved is not None
    assert isclose(moved[0] - base[0], 0.2 * FRAME[0], abs_tol=6.0)


def test_size_is_scale_times_partial_bbox():
    pts = _upright_hand()
    px = [(x * FRAME[0], y * FRAME[1]) for x, y in pts]
    pxs = [px[i][0] for i in PARTIAL]
    pys = [px[i][1] for i in PARTIAL]
    expected = TRACK_SCALE * max(max(pxs) - min(pxs), max(pys) - min(pys))
    rect = rect_from_landmarks(pts, *FRAME)
    assert rect is not None
    assert isclose(rect[2], expected, rel_tol=0.03), (rect[2], expected)


def test_shift_moves_center_toward_fingertips():
    pts = _upright_hand()
    px = [(x * FRAME[0], y * FRAME[1]) for x, y in pts]
    pys = [px[i][1] for i in PARTIAL]
    partial_cy = (max(pys) + min(pys)) / 2.0
    rect = rect_from_landmarks(pts, *FRAME)
    assert rect is not None
    # shift_y=-0.1 lifts the center UP (y-down image coords) toward fingers.
    assert rect[1] < partial_cy


def test_all_landmarks_inside_crop():
    """Inverse of _rotate_and_crop: every landmark must land in [0, size]."""
    for deg in (0, 30, -45):
        pts = _rotate(_upright_hand(), deg)
        rect = rect_from_landmarks(pts, *FRAME)
        assert rect is not None
        cx, cy, size = rect[0], rect[1], rect[2]
        a = radians(rect[4])
        ca, sa = cos(a), sin(a)
        for x, y in pts:
            fx, fy = x * FRAME[0], y * FRAME[1]
            ux, uy = fx - cx, fy - cy
            # frame -> crop: rotate by -angle, then shift by size/2
            crop_x = (ux * ca + uy * sa) + size / 2
            crop_y = (-ux * sa + uy * ca) + size / 2
            assert -2 <= crop_x <= size + 2, (deg, crop_x, size)
            assert -2 <= crop_y <= size + 2, (deg, crop_y, size)


def test_degenerate_hand_returns_none():
    pts = [(0.5, 0.5)] * 21
    assert rect_from_landmarks(pts, *FRAME) is None
