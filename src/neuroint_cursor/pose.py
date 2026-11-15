"""Pointing pose: index finger raised, the rest curled (the "pointing" pose).

Rotation-invariant: a finger counts as straight when its tip is farther
from the wrist than its PIP joint AND the tip-pip-mcp chain is not bent.
An open palm does NOT count (all fingers extended) — only the pointing
pose engages cursor control.
"""

from __future__ import annotations

from math import acos, degrees, sqrt

WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

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


def is_pointing(pts: list[tuple[float, float]]) -> bool:
    """The pointing pose: index straight, middle folded, index the extremum.

    Middle alone separates ☝ from ✌ and the open palm. Ring/pinky are
    NOT state-checked: on a hand seen edge-on (knuckles toward the
    camera, the dark-hoodie case) they misread as extended — the face
    false positives that motivated the strict rule are already rejected
    by the palm-cluster filter before this check runs.

    Third condition — the physical invariant of ☝: the index TIP is the
    farthest fingertip from the wrist (it is the extended one). Junk
    hallucinations that pass the two state checks almost always put
    some other fingertip farther out; a real pointing hand never does
    (0.85 tolerance for edge-on hands where a folded-tip estimate can
    creep slightly past).
    """
    state = finger_states(pts)
    if not (state["index"] and not state["middle"]):
        return False
    di = _dist(pts[WRIST], pts[INDEX_TIP])
    farthest = max(
        _dist(pts[WRIST], pts[t]) for t in (MIDDLE_TIP, RING_TIP, PINKY_TIP)
    )
    return di >= 0.85 * farthest


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
