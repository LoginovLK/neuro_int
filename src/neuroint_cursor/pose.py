"""Hand poses: the industry-standard gesture vocabulary.

Scheme (Ultraleap / Vision Pro / camera-mouse convention), all checks
scale-free in units of the palm length (wrist -> middle MCP):

  open hand            -> MOVE  (cursor follows the palm center)
  pinch thumb-index    -> LMB press, release on open (click / hold-drag)
  index+middle (V)     -> SCROLL: vertical hand travel = wheel steps
  fist / bundle (held) -> REST "mouse down": cursor parked (is_fist_bundle,
                          the exact anti-pinch signature — a frame that
                          could click can never park, and vice versa)
  (RMB pinch removed: this user's bundle is indistinguishable from a
   thumb-middle pinch on raw distances — 15/15 false fires, see below)

Clicks and drags come from thumb-to-fingertip DISTANCES with hysteresis,
never from silhouette classification - a distance cannot be faked by
hand tilt (the old 2D-projection counts misread tilted palms as clicks).
"""

from __future__ import annotations

from math import acos, degrees, sqrt

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

TIP_IDS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# fist vs pinch separators: the thumb sits ON the index phalanges in a
# fist / gathered bundle (thumb-tip to index-PIP ~= thumb-tip to index-TIP)
# and a phalanx away from the PIP in a true pinch
FIST_GUARD_PIP = 0.30
FIST_GUARD_DELTA = 0.10

FINGERS: dict[str, tuple[int, int, int]] = {
    "index": (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    "ring": (RING_TIP, RING_PIP, RING_MCP),
    "pinky": (PINKY_TIP, PINKY_PIP, PINKY_MCP),
}


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def palm_length(pts: list[tuple[float, float]]) -> float:
    """Wrist -> middle MCP: the scale unit for every threshold."""
    return _dist(pts[WRIST], pts[MIDDLE_MCP])


# the cursor control point: mean of the wrist and the four knuckle roots.
# Knuckles barely move when fingers flex (bending happens at PIP/DIP),
# so a pinch does not drag the cursor, and the 5-bone mean is far more
# stable under wrist rotation than any single landmark — a single index
# knuckle visibly slides when the hand tilts (measured live).
PALM_IDS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


def palm_center(pts: list[tuple[float, float]]) -> tuple[float, float]:
    x = sum(pts[i][0] for i in PALM_IDS) / len(PALM_IDS)
    y = sum(pts[i][1] for i in PALM_IDS) / len(PALM_IDS)
    return x, y


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
    """Center of the four fingertips (the old bundle point; coast velocity)."""
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


def hull_ratio(pts: list[tuple[float, float]]) -> float:
    """Convex-hull area of all 21 points, in palm^2 units."""
    hand = palm_length(pts)
    if hand < 1e-6:
        return 0.0
    return _hull_area(pts) / (hand * hand)


def is_pinch(
    pts: list[tuple[float, float]],
    max_hull: float = 1.0,
    min_reach: float = 0.6,
    max_reach: float = 1.9,
) -> bool:
    """The legacy BUNDLE pose (four tips gathered, hand stretched forward).

    Not a click: it is one of the compact REST shapes. Kept for the ONNX
    fallback backend's candidate bonus / quarantine logic.
    """
    hand = palm_length(pts)
    if hand < 1e-6:
        return False
    cx, cy = pinch_center(pts)
    reach = _dist(pts[WRIST], (cx, cy)) / hand
    if not (min_reach <= reach <= max_reach):
        return False
    return _hull_area(pts) <= max_hull * hand * hand


def pinch_dist(
    pts: list[tuple[float, float]], finger: str = "index", palm: float | None = None
) -> float:
    """Thumb-tip to fingertip distance in palm-length units.

    `palm` substitutes a SMOOTHED palm length: dividing by the instant
    per-frame palm makes every distance jump when the hand turns
    edge-on (the projected wrist->MCP collapses and all ratios blow up
    2-4x — measured live); a slow EMA denominator keeps the units
    stable for the lifetime of a gesture.
    """
    hand = palm if palm is not None else palm_length(pts)
    if hand is None or hand < 1e-6:
        return 9.9
    tip = {"index": INDEX_TIP, "middle": MIDDLE_TIP}[finger]
    return _dist(pts[THUMB_TIP], pts[tip]) / hand


def thumb_pip_dist(pts: list[tuple[float, float]], palm: float | None = None) -> float:
    """Thumb-tip to index-PIP distance, palm units — the fist discriminator.

    A pinch brings the thumb to the index TIP, a full phalanx away from
    the PIP; a fist wraps the thumb OVER the index phalanges, right at
    the PIP. This one distance separates the two compact hand shapes
    that defeated every silhouette-based guard.
    """
    hand = palm if palm is not None else palm_length(pts)
    if hand is None or hand < 1e-6:
        return 9.9
    return _dist(pts[THUMB_TIP], pts[INDEX_PIP]) / hand


def is_rest_pose(pts: list[tuple[float, float]]) -> bool:
    """Fist / bundle: no extended fingers and a compact hull."""
    if any(finger_states(pts).values()):
        return False
    return hull_ratio(pts) <= 1.0


def is_scroll_pose(pts: list[tuple[float, float]]) -> bool:
    """Victory: index+middle extended, ring+pinky folded (trackpad scroll).

    The two raised fingers only need to be UP (bend <= 60 degrees): the
    strict 50-degree straightness flickered on a relaxed V and chattered
    scroll on/off within 100ms (measured live). Ring+pinky must stay
    folded, so an open palm never reads as scroll.
    """
    up = finger_extended(pts, *FINGERS["index"], max_bend=60.0) and (
        finger_extended(pts, *FINGERS["middle"], max_bend=60.0)
    )
    if not up:
        return False
    st = finger_states(pts)
    return not st["ring"] and not st["pinky"]


def is_move_pose(pts: list[tuple[float, float]]) -> bool:
    """Anything open and not scroll/rest: the cursor-driving pose."""
    if is_rest_pose(pts) or is_scroll_pose(pts):
        return False
    return any(finger_states(pts).values()) or hull_ratio(pts) > 1.0


def pinch_l_fire(d48: float, d46: float, close_dist: float) -> bool:
    """The LMB press gate on precomputed distances (median-filtered
    upstream): tips closed AND the thumb at the TIP, not the phalanges."""
    if d48 >= close_dist:
        return False
    return d46 > FIST_GUARD_PIP or (d46 - d48) > FIST_GUARD_DELTA


def is_fist_bundle(d48: float, d46: float) -> bool:
    """The REST signature — the exact negation of the pinch gate: the
    thumb sits ON the index phalanges (d46 small) with no tip gap. A
    frame that could fire a pinch can never count toward rest and vice
    versa, so parking the cursor can never eat a click."""
    return d46 < FIST_GUARD_PIP and (d46 - d48) < FIST_GUARD_DELTA


def pinch_l_valid(
    pts: list[tuple[float, float]], close_dist: float, palm: float | None = None
) -> bool:
    """LMB pinch guard: thumb-index TIPS closed AND the thumb is at the
    TIP, not on the index phalanges (fist / gathered bundle). Measured on
    real frames: a bundle puts the thumb near BOTH the tip and the PIP
    (d46 ~= d48 ~= 0.24), a fist puts it ON the PIP (d46 << d48), a true
    pinch has the tip a phalanx farther (d46 > d48 by a clear margin).
    Relaxed other fingers no longer matter — the click does not depend
    on finger styling.
    """
    return pinch_l_fire(
        pinch_dist(pts, "index", palm), thumb_pip_dist(pts, palm), close_dist
    )


# RMB pinch gesture REMOVED (measured live, 2026-11-22): the user's
# gathered bundle runs d412 BELOW d48 (0.07 vs 0.14-0.20), so "middle
# finger dominance" fired 15/15 false RMBs inside bundles and pinch
# transitions. For this hand no threshold separates a thumb-middle pinch
# from an index-pinch bundle — the gesture is not implementable on raw
# distances and stays out until a distinguishable signal exists.


def classify(pts: list[tuple[float, float]]) -> str:
    """Coarse shape class (shape-lock refit trigger only, not clicks)."""
    if is_rest_pose(pts):
        return "rest"
    if is_scroll_pose(pts):
        return "scroll"
    if is_move_pose(pts):
        return "move"
    return "other"


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
