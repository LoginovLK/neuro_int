"""Bundle (pinch) pose: synthetic hands, rotation invariance.

The pinch gates (all scale-free, units of the palm length wrist->
middle-MCP): tip-cluster radius <= 0.32, bundle reach >= 0.9 from the
wrist, and >= 3 fingers with the tip farther from the wrist than the
PIP (a fist folds tips back onto the knuckles and fails this).
"""

from math import cos, radians, sin

from neuroint_cursor.pose import is_pinch, pinch_center

WRIST = (0.50, 0.85)


def _pinch_finger(mcp, bundle, tip_offset):
    """mcp -> pip -> dip -> tip converging into the bundle point."""
    bx, by = bundle
    return [
        mcp,
        (mcp[0] + (bx - mcp[0]) * 0.45, mcp[1] + (by - mcp[1]) * 0.45),
        (mcp[0] + (bx - mcp[0]) * 0.80, mcp[1] + (by - mcp[1]) * 0.80),
        (bx + tip_offset[0], by + tip_offset[1]),
    ]


def _straight_finger(mcp, length):
    """mcp -> pip -> dip -> tip straight up."""
    return [
        mcp,
        (mcp[0] + 0.002, mcp[1] - length * 0.4),
        (mcp[0] + 0.001, mcp[1] - length * 0.7),
        (mcp[0], mcp[1] - length),
    ]


def _curled_finger(mcp, length):
    """mcp -> pip up, then tip folded back down to the knuckle line."""
    return [
        mcp,
        (mcp[0] + 0.002, mcp[1] - length * 0.55),
        (mcp[0] + 0.004, mcp[1] - length * 0.30),
        (mcp[0] + 0.008, mcp[1] - length * 0.02),
    ]


def _hand(fingers):
    pts = [WRIST]
    pts += [(0.42, 0.80), (0.41, 0.74), (0.42, 0.68), (0.44, 0.62)]  # thumb 1-4
    for f in fingers:
        pts += f
    assert len(pts) == 21
    return pts


PINCH_BUNDLE = (0.50, 0.53)
MCP_INDEX, MCP_MIDDLE, MCP_RING, MCP_PINKY = (
    (0.46, 0.68),
    (0.50, 0.66),
    (0.54, 0.67),
    (0.58, 0.69),
)


def _pinch_hand():
    tips = [(-0.008, 0.004), (0.002, -0.006), (0.006, 0.002), (0.010, 0.006)]
    mcps = (MCP_INDEX, MCP_MIDDLE, MCP_RING, MCP_PINKY)
    return _hand(
        [_pinch_finger(mcp, PINCH_BUNDLE, off) for mcp, off in zip(mcps, tips, strict=True)]
    )


def _open_palm_hand():
    return _hand(
        [
            _straight_finger(mcp, 0.28)
            for mcp in (MCP_INDEX, MCP_MIDDLE, MCP_RING, MCP_PINKY)
        ]
    )


def _fist_hand():
    return _hand(
        [
            _curled_finger(mcp, 0.20)
            for mcp in (MCP_INDEX, MCP_MIDDLE, MCP_RING, MCP_PINKY)
        ]
    )


def _peace_hand():
    return _hand(
        [
            _straight_finger(MCP_INDEX, 0.28),
            _straight_finger(MCP_MIDDLE, 0.28),
            _curled_finger(MCP_RING, 0.20),
            _curled_finger(MCP_PINKY, 0.20),
        ]
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


def test_pinch_detected():
    assert is_pinch(_pinch_hand())


def test_open_palm_is_not_pinch():
    assert not is_pinch(_open_palm_hand())


def test_fist_is_not_pinch():
    """Fist: tips fold back onto the knuckles — tip closer than pip."""
    assert not is_pinch(_fist_hand())


def test_peace_is_not_pinch():
    assert not is_pinch(_peace_hand())


def test_pinch_center_is_the_bundle():
    cx, cy = pinch_center(_pinch_hand())
    assert abs(cx - PINCH_BUNDLE[0]) < 0.02
    assert abs(cy - PINCH_BUNDLE[1]) < 0.02


def test_pinch_is_rotation_invariant():
    pts = _pinch_hand()
    for deg in (40, -70, 180):
        assert is_pinch(_rotate(pts, deg)), f"failed at {deg} deg"


def test_open_palm_stays_not_pinch_when_rotated():
    pts = _open_palm_hand()
    for deg in (30, -50, 160):
        assert not is_pinch(_rotate(pts, deg)), f"failed at {deg} deg"
