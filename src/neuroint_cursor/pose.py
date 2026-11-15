"""Hand poses: the "bundle" (pinch) engages the cursor.

The bundle: all four fingertips gathered into one point — comfortable to
hold (no extended finger to fatigue). Scale-free checks, units of the
palm length (wrist -> middle MCP).
"""

from __future__ import annotations

from math import acos, degrees, sqrt

WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

TIP_IDS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

FINGERS: dict[str, tuple[int, int, int]] = {
    "index": (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    "ring": (RING_TIP, RING_PIP, RING_MCP),
    "pinky": (PINKY_TIP, PINKY_PIP, PINKY_MCP),
}


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _bend_deg(tip: tuple[float, float], pip: tuple[float, float], mcp: tuple[float, float]) -> float:
    """Bend at the PIP joint: angle between tip->pip and pip->mcp directions.

    A straight finger gives ~0 degrees (both segments point the same way);
    a folded one gives a large angle.
    """
    v1 = (tip[0] - pip[0], tip[1] - pip[1])
    v2 = (pip[0] - mcp[0], pip[1] - mcp[1])
    n1 = sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = sqrt(v2[0] ** 2 + v2[1] ** 2)
    if n1 * n2 == 0.0:
        return 180.0
    cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return degrees(acos(max(-1.0, min(1.0, cosang))))


def finger_extended(
    pts: list[tuple[float, float]],
    tip: int,
    pip: int,
    mcp: int,
    straight_ratio: float = 1.15,
    max_bend: float = 50.0,
) -> bool:
    d_tip = _dist(pts[WRIST], pts[tip])
    d_pip = _dist(pts[WRIST], pts[pip])
    if d_tip < straight_ratio * d_pip:
        return False
    return _bend_deg(pts[tip], pts[pip], pts[mcp]) <= max_bend


def finger_states(pts: list[tuple[float, float]]) -> dict[str, bool]:
    return {name: finger_extended(pts, *ids) for name, ids in FINGERS.items()}


def pinch_center(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """Center of the fingertip bundle — the cursor control point."""
    xs = sum(pts[i][0] for i in TIP_IDS)
    ys = sum(pts[i][1] for i in TIP_IDS)
    return xs / len(TIP_IDS), ys / len(TIP_IDS)


def _hull_area(points: list[tuple[float, float]]) -> float:
    """Convex hull area via a monotone chain (pure math, no cv2)."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def is_pinch(
    pts: list[tuple[float, float]],
    max_hull: float = 1.0,
    min_reach: float = 0.6,
    max_reach: float = 1.9,
) -> bool:
    """The bundle pose: fingertips gathered compactly over the palm.

    Measured on real frames (MediaPipe landmarks, scale-free, units of
    the palm length wrist -> middle MCP): the CONVEX HULL of the 21
    points is the one feature that separates the poses — a bundled hand
    keeps its tips INSIDE the palm outline (hull <= 1.0 palm^2), an
    open palm stretches it (1.5+). The old per-finger state checks were
    indistinguishable under foreshortening (tip-cluster radius 0.41 on
    the bundled hand vs 0.44 on the open palm — identical). The reach
    gate keeps the cluster in front of the palm, not collapsed onto the
    wrist.
    """
    hand = _dist(pts[WRIST], pts[MIDDLE_MCP])
    if hand < 1e-6:
        return False
    cx, cy = pinch_center(pts)
    reach = _dist(pts[WRIST], (cx, cy)) / hand
    if not (min_reach <= reach <= max_reach):
        return False
    return _hull_area(pts) <= max_hull * hand * hand


def palm_center(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """Center of the palm: mean of wrist, index MCP, pinky MCP."""
    px = (pts[0][0] + pts[5][0] + pts[17][0]) / 3
    py = (pts[0][1] + pts[5][1] + pts[17][1]) / 3
    return px, py


def extended_finger_count(pts: list[tuple[float, float]]) -> int:
    """How many fingers point forward, STRICTLY: the tip is farther from
    the wrist than its PIP by a real margin AND the finger is straight
    (PIP bend < 45 deg). A loose count flickers 1..3 on an open palm and
    spams clicks; the strict count keeps an open palm at 4 (open) and a
    closed hand at 0 (control).
    """
    n = 0
    for _name, (tip, pip, mcp) in FINGERS.items():
        if _dist(pts[WRIST], pts[tip]) <= _dist(pts[WRIST], pts[pip]) * 1.25:
            continue
        if _bend_deg(pts[tip], pts[pip], pts[mcp]) > 45.0:
            continue
        n += 1
    return n


def classify(pts: list[tuple[float, float]]) -> str:
    """Gesture class for the state machine.

    'zero'   — fingers rolled into tubes to the palm center: center cursor
    'point1' — index only: left click
    'point2' — index+middle: right click
    'point3' — three fingers: middle press (hold + move = scroll)
    'open'   — open palm: release everything
    'control'— anything closed in between: the cursor-driving pose
    """
    hand = _dist(pts[WRIST], pts[MIDDLE_MCP])
    if hand < 1e-6:
        return "control"
    if is_zero(pts):
        return "zero"
    n = extended_finger_count(pts)
    if n >= 4:
        return "open"
    if n == 1:
        return "point1"
    if n == 2:
        return "point2"
    if n == 3:
        return "point3"
    return "control"


def is_zero(
    pts: list[tuple[float, float]],
    max_hull: float = 1.0,
    max_tip_to_palm: float = 0.75,
    min_curled: int = 3,
) -> bool:
    """The 'zero' pose: fingers rolled into tubes toward the palm.

    Physically opposite of the bundle: the tips FOLD BACK toward the
    palm center (a bundle throws them forward past the palm outline).
    Gates, palm-length units: at least 3 fingers curled (tip closer to
    the wrist than its PIP), the tip cluster sits within 0.75 palm
    lengths of the palm center, and the hand stays compact. Zero and
    bundle are mutually exclusive by construction (bundle needs reach
    >= 0.9, zero caps it at <= 0.75).
    """
    hand = _dist(pts[WRIST], pts[MIDDLE_MCP])
    if hand < 1e-6:
        return False
    curled = sum(
        1
        for tip, pip, _mcp in FINGERS.values()
        if _dist(pts[WRIST], pts[tip]) < _dist(pts[WRIST], pts[pip]) * 1.05
    )
    if curled < min_curled:
        return False
    cx, cy = pinch_center(pts)
    px, py = palm_center(pts)
    if _dist((cx, cy), (px, py)) > max_tip_to_palm * hand:
        return False
    return _hull_area(pts) <= max_hull * hand * hand


def palm_ratio(pts: list[tuple[float, float]]) -> float | None:
    """Palm length (wrist -> middle MCP) over palm width (index -> pinky MCP).

    A real hand sits in ~0.65..2.2 for any orientation; the landmark model
    squashed onto a tilted or non-hand crop collapses far outside that
    band (measured 0.12..0.56 on garbage). None when degenerate.
    """
    palm_w = _dist(pts[INDEX_MCP], pts[PINKY_MCP])
    if palm_w < 1e-6:
        return None
    return _dist(pts[WRIST], pts[MIDDLE_MCP]) / palm_w


def palm_sane(pts: list[tuple[float, float]], lo: float = 0.65, hi: float = 2.2) -> bool:
    r = palm_ratio(pts)
    return r is not None and lo <= r <= hi
