"""Shared pytest fixtures/helpers for gesture pipeline tests."""

from __future__ import annotations

from pathlib import Path

from neuroint_protocol.gestures import GestureEvent

REPO_ROOT = Path(__file__).resolve().parents[1]


def feed_frames(
    fsm, hand_factory, count: int, dt_ms: int = 33, start_ms: int = 0
) -> list[GestureEvent]:
    """Feed ``count`` consecutive frames into the FSM, return emitted events."""
    events: list[GestureEvent] = []
    t = start_ms
    for _ in range(count):
        hand = hand_factory(t)
        events.extend(fsm.feed(hand))
        t += dt_ms
    return events
