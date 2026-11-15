"""Pointing-pose logic: synthetic hands, rotation invariance."""

from math import cos, radians, sin

from neuroint_cursor.pose import finger_states, is_pointing

WRIST = (0.50, 0.90)


def _straight_finger(mcp_x: float, base_y: float, up: float = 0.30):
    """mcp -> pip -> dip -> tip going straight up (decreasing y)."""
    return [
        (mcp_x, base_y),                    # mcp
        (mcp_x + 0.01, base_y - up * 0.4),  # pip
        (mcp_x + 0.008, base_y - up * 0.7), # dip
        (mcp_x + 0.005, base_y - up),       # tip
    ]


def _curled_finger(mcp_x: float, base_y: float, up: float = 0.24):
    """mcp -> pip going up, then tip folded back down (folded at the PIP)."""
    return [
        (mcp_x, base_y),                     # mcp
        (mcp_x + 0.01, base_y - up * 0.55),  # pip (up)
        (mcp_x + 0.02, base_y - up * 0.50),  # dip
        (mcp_x + 0.04, base_y - up * 0.05),  # tip (folded back down)
    ]


def _hand(index, middle, ring, pinky):
    pts = [WRIST]
    pts += [(0.40, 0.84), (0.38, 0.80), (0.37, 0.77), (0.36, 0.74)]  # thumb 1-4
    for f in (index, middle, ring, pinky):
        pts += f
    assert len(pts) == 21
    return pts


def _pointing_hand():
    return _hand(
        _straight_finger(0.47, 0.70),
        _curled_finger(0.52, 0.68),
        _curled_finger(0.56, 0.70),
        _curled_finger(0.60, 0.72),
    )


def _open_palm_hand():
    return _hand(
        _straight_finger(0.47, 0.70),
        _straight_finger(0.52, 0.68),
        _straight_finger(0.56, 0.70),
        _straight_finger(0.60, 0.72),
    )


def _fist_hand():
    return _hand(
        _curled_finger(0.47, 0.70),
        _curled_finger(0.52, 0.68),
        _curled_finger(0.56, 0.70),
        _curled_finger(0.60, 0.72),
    )


def _rotate(pts, deg):
    a = radians(deg)
    ca, sa = cos(a), sin(a)
    wx, wy = WRIST
    out = []
    for x, y in pts:
        dx, dy = x - wx, y - wy
        out.append((wx + dx * ca - dy * sa, wy + dx * sa + dy * ca))
    return out


def test_pointing_hand_detected():
    assert is_pointing(_pointing_hand())


def test_open_palm_is_not_pointing():
    state = finger_states(_open_palm_hand())
    assert state["index"] and state["middle"]  # sanity: palm really is open
    assert not is_pointing(_open_palm_hand())


def test_fist_is_not_pointing():
    state = finger_states(_fist_hand())
    assert not any(state.values())
    assert not is_pointing(_fist_hand())


def test_sloppy_ring_is_still_pointing():
    """Mid-motion ring/pinky may read extended — still pointing."""
    pts = _hand(
        _straight_finger(0.47, 0.70),
        _curled_finger(0.52, 0.68),
        _straight_finger(0.56, 0.70),  # sloppy ring
        _curled_finger(0.60, 0.72),
    )
    assert is_pointing(pts)


def test_sloppy_ring_and_pinky_still_pointing():
    """Edge-on hands (knuckles to camera) misread ring/pinky as extended;
    middle folded is what keeps ☝ distinct from ✌ and the palm."""
    pts = _hand(
        _straight_finger(0.47, 0.70),
        _curled_finger(0.52, 0.68),
        _straight_finger(0.56, 0.70),
        _straight_finger(0.60, 0.72),
    )
    assert is_pointing(pts)


def test_peace_is_not_pointing():
    """index + middle up is ✌, not ☝."""
    pts = _hand(
        _straight_finger(0.47, 0.70),
        _straight_finger(0.52, 0.68),
        _curled_finger(0.56, 0.70),
        _curled_finger(0.60, 0.72),
    )
    assert not is_pointing(pts)


def test_index_must_be_the_extremum():
    """☝ physical invariant: the index TIP is the farthest fingertip from
    the wrist. Junk hallucinations passing the state checks put some
    other fingertip farther out — they must not read as pointing."""
    pts = _hand(
        _straight_finger(0.47, 0.70),
        _curled_finger(0.52, 0.68),
        _curled_finger(0.56, 0.70),
        _straight_finger(0.60, 0.72, up=0.45),  # pinky stretched way out
    )
    assert not is_pointing(pts)


def test_pointing_is_rotation_invariant():
    pts = _pointing_hand()
    for deg in (40, -70, 180):
        assert is_pointing(_rotate(pts, deg)), f"failed at {deg} deg"


def test_open_palm_stays_not_pointing_when_rotated():
    pts = _open_palm_hand()
    for deg in (30, -50, 160):
        assert not is_pointing(_rotate(pts, deg)), f"failed at {deg} deg"
