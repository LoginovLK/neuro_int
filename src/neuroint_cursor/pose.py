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
    """The pointing pose, tolerant of mid-motion sloppiness.

    Index must be straight and the middle finger folded (that alone
    separates ☝ from ✌ and the open palm). Ring/pinky may read sloppy
    while the hand moves — only BOTH extended counts as palm-like.
    """
    state = finger_states(pts)
    if not state["index"] or state["middle"]:
        return False
    return not (state["ring"] and state["pinky"])
