"""Dynamic gesture recognition as an explicit finite state machine.

Swipe/hold flow:
    IDLE  --pose held arm_hold_ms-->  ARMED
    ARMED --tip moves swipe_min_distance in swipe_window_ms--> SWIPE -> emit -> COOLDOWN
    ARMED --pose changes / hand lost--> IDLE
    any hold-only pose kept kill_hold_ms --> emit hold event -> COOLDOWN

Cursor flow (pose = pinch):
    IDLE --pinch seen--> CURSOR (edge drives the mouse from cursor coords)
    CURSOR --pinch returns after a short gap (< pending_click_ms)--> click event
    CURSOR --pinch lost / other pose / hand gone--> CURSOR_PENDING -> IDLE

While in CURSOR, swipe gestures are suppressed: the hand that steers the
mouse must not also fire Alt+Tab. Kill switch (open_palm hold) still works
because leaving cursor mode re-arms the normal flow.

Distances are measured in palm widths so thresholds hold regardless of how
far the user sits from the camera.
"""

from __future__ import annotations

import enum
import logging
import time

from neuroint_protocol.gestures import Direction, GestureEvent, Pose

from neuroint_brain.gestures.poses import classify_pose
from neuroint_brain.vision.hands import INDEX_TIP, THUMB_TIP, HandObservation
from neuroint_brain.vision.smoothing import PointFilter2D

logger = logging.getLogger(__name__)

SWIPE_POSES = frozenset({Pose.INDEX_UP, Pose.TWO_UP})
HOLD_POSES = frozenset({Pose.OPEN_PALM, Pose.FIST})


class _State(enum.Enum):
    IDLE = "idle"
    ARMED = "armed"
    COOLDOWN = "cooldown"
    CURSOR = "cursor"
    CURSOR_PENDING = "cursor_pending"
    HOLD_LOCK = "hold_lock"


class GestureFSM:
    """Consumes per-frame hand observations, emits completed gesture events.

    Live-UI snapshot properties: :attr:`state_name`, :attr:`last_pose`,
    :attr:`cursor_active`, :attr:`cursor_x`, :attr:`cursor_y`.
    """

    def __init__(
        self,
        arm_hold_ms: int = 220,
        swipe_window_ms: int = 600,
        swipe_min_distance: float = 2.2,
        hand_lost_reset_ms: int = 350,
        cooldown_ms: int = 650,
        kill_hold_ms: int = 1100,
        pending_click_ms: int = 380,
        click_rearm_ms: int = 450,
        cursor_lost_ms: int = 600,
    ) -> None:
        self.arm_hold_ms = arm_hold_ms
        self.swipe_window_ms = swipe_window_ms
        self.swipe_min_distance = swipe_min_distance
        self.hand_lost_reset_ms = hand_lost_reset_ms
        self.cooldown_ms = cooldown_ms
        self.kill_hold_ms = kill_hold_ms
        # Cursor tuning: a quick pinch-release-pinch inside pending_click_ms
        # emits a click; click_rearm_ms guards against accidental doubles.
        # A hand lost (motion blur) for less than cursor_lost_ms does NOT
        # exit cursor mode — fast movements must keep steering the mouse.
        self.pending_click_ms = pending_click_ms
        self.click_rearm_ms = click_rearm_ms
        self.cursor_lost_ms = cursor_lost_ms

        self._state = _State.IDLE
        self._pose: Pose = Pose.UNKNOWN
        self._candidate_pose: Pose = Pose.UNKNOWN
        self._candidate_since: float = 0.0
        self._armed_since: float = 0.0
        self._anchor: tuple[float, float] = (0.0, 0.0)
        self._last_seen: float = 0.0
        self._cooldown_until: float = 0.0
        self._hold_emitted: bool = False
        self._lock_pose: Pose = Pose.UNKNOWN
        self._tip_filter = PointFilter2D(min_cutoff=1.4, beta=0.02)
        self._cursor_filter = PointFilter2D(min_cutoff=1.0, beta=0.01)

        # Live-UI snapshot
        self._pending_since: float = 0.0
        self._pending_explicit = False  # pinch was released to a RECOGNIZED pose
        self._next_click_at: float = 0.0
        self._cursor_active = False
        self._cursor_x = 0.0
        self._cursor_y = 0.0
        self._last_pose: Pose = Pose.UNKNOWN

    # --- live-UI snapshot ---------------------------------------------------

    @property
    def state_name(self) -> str:
        return self._state.value

    @property
    def last_pose(self) -> Pose:
        return self._last_pose

    @property
    def cursor_active(self) -> bool:
        return self._cursor_active

    @property
    def cursor_x(self) -> float:
        return self._cursor_x

    @property
    def cursor_y(self) -> float:
        return self._cursor_y

    # --- core ----------------------------------------------------------------

    def reset(self) -> None:
        self._state = _State.IDLE
        self._pose = Pose.UNKNOWN
        self._candidate_pose = Pose.UNKNOWN
        self._hold_emitted = False
        self._cursor_active = False

    def feed(self, hand: HandObservation | None) -> list[GestureEvent]:
        """Feed one frame; returns zero, one or more completed events."""
        now_s: float = hand.timestamp_ms / 1000.0 if hand else time.monotonic()

        if hand is None:
            self._last_pose = Pose.UNKNOWN
            if self._state == _State.HOLD_LOCK:
                # Hand disappeared = the hold was physically released.
                self._state = _State.IDLE
                self._candidate_pose = Pose.UNKNOWN
                return []
            if self._state == _State.CURSOR:
                # Motion-blur frames lose the hand entirely; hold cursor mode
                # open for cursor_lost_ms before giving up.
                self._state = _State.CURSOR_PENDING
                self._pending_since = now_s
                self._pending_explicit = False
                self._cursor_active = False
                return []
            if self._state == _State.CURSOR_PENDING:
                if (now_s - self._pending_since) * 1000 > self.cursor_lost_ms:
                    return self._leave_cursor(now_s, hand_gone=True)
                return []
            if self._state == _State.ARMED and (
                (now_s - self._last_seen) * 1000 > self.hand_lost_reset_ms
            ):
                logger.debug("hand lost while armed -> reset")
                self.reset()
            return []

        self._last_seen = now_s
        pose = classify_pose(hand)
        self._last_pose = pose
        tip = self._tip_filter.filter(*hand.fingertip(), timestamp_s=now_s)
        cursor_point = self._cursor_point(hand, now_s)

        if self._state in (_State.CURSOR, _State.CURSOR_PENDING):
            return self._feed_cursor(pose, cursor_point, now_s)

        if self._state == _State.HOLD_LOCK:
            # Hold already fired once; wait for the pose to change (the
            # user opening/closing the hand again) before re-arming.
            if pose != self._lock_pose:
                self._state = _State.IDLE
                self._candidate_pose = Pose.UNKNOWN
                self._candidate_since = now_s
            return []

        if self._state == _State.COOLDOWN:
            if now_s < self._cooldown_until:
                return []
            self._state = _State.IDLE

        if self._state == _State.IDLE:
            if pose == Pose.PINCH:
                # Intentional grab: enter cursor mode without the arm delay.
                self._enter_cursor(now_s, cursor_point)
                return []
            if pose in SWIPE_POSES or pose in HOLD_POSES:
                # Pose must persist for arm_hold_ms before the FSM arms:
                # a hand casually passing through the frame must not trigger.
                if pose != self._candidate_pose:
                    self._candidate_pose = pose
                    self._candidate_since = now_s
                elif (now_s - self._candidate_since) * 1000 >= self.arm_hold_ms:
                    self._state = _State.ARMED
                    self._pose = pose
                    self._armed_since = now_s
                    self._anchor = tip
                    self._hold_emitted = False
            else:
                self._candidate_pose = Pose.UNKNOWN
            return []

        # --- ARMED ---
        if pose != self._pose:
            logger.debug("pose changed %s -> %s while armed, reset", self._pose, pose)
            self.reset()
            if pose == Pose.PINCH:
                self._enter_cursor(now_s, cursor_point)
            return []

        elapsed_ms = (now_s - self._armed_since) * 1000

        if pose in HOLD_POSES:
            if not self._hold_emitted and elapsed_ms >= self.kill_hold_ms:
                self._hold_emitted = True
                events = [GestureEvent(pose=pose, direction=Direction.NONE)]
                # One shot per physical hold: no re-trigger while held.
                self._state = _State.HOLD_LOCK
                self._lock_pose = pose
                return events
            return []

        # Swipe detection in palm-width units.
        palm = hand.palm_size()
        if palm < 1e-4:  # degenerate detection, ignore frame
            return []
        dx = (tip[0] - self._anchor[0]) / palm
        dy = (tip[1] - self._anchor[1]) / palm

        if elapsed_ms <= self.swipe_window_ms:
            distance = (dx * dx + dy * dy) ** 0.5
            if distance >= self.swipe_min_distance:
                direction = _dominant_direction(dx, dy)
                if direction != Direction.NONE:
                    self._start_cooldown(now_s)
                    return [GestureEvent(pose=pose, direction=direction)]
        else:
            # Window expired without a decisive motion: re-anchor so a slow
            # drift does not eventually fake a swipe.
            self._armed_since = now_s
            self._anchor = tip
        return []

    # --- cursor internals ------------------------------------------------------

    def _cursor_point(self, hand: HandObservation, now_s: float) -> tuple[float, float]:
        """Midpoint of thumb and index tips: the steering point for the mouse."""
        thumb = hand.landmarks[THUMB_TIP]
        index = hand.landmarks[INDEX_TIP]
        return self._cursor_filter.filter(
            (thumb[0] + index[0]) / 2.0,
            (thumb[1] + index[1]) / 2.0,
            timestamp_s=now_s,
        )

    def _enter_cursor(self, now_s: float, cursor_point: tuple[float, float]) -> None:
        self._state = _State.CURSOR
        self._cursor_active = True
        self._cursor_x, self._cursor_y = cursor_point
        logger.info("cursor mode ON")

    def _feed_cursor(
        self, pose: Pose, cursor_point: tuple[float, float], now_s: float
    ) -> list[GestureEvent]:
        if pose == Pose.PINCH:
            events: list[GestureEvent] = []
            if self._state == _State.CURSOR_PENDING:
                gap_ms = max(0.0, (now_s - self._pending_since) * 1000)
                self._state = _State.CURSOR
                # A click requires an explicit release to a recognized pose:
                # blur/unrecognized frames just resume steering.
                if (
                    self._pending_explicit
                    and gap_ms <= self.pending_click_ms
                    and now_s >= self._next_click_at
                ):
                    self._next_click_at = now_s + self.click_rearm_ms / 1000.0
                    events.append(GestureEvent(pose=Pose.PINCH, direction=Direction.NONE))
                    logger.debug("cursor click")
            self._pending_explicit = False
            self._cursor_active = True
            self._cursor_x, self._cursor_y = cursor_point
            return events

        # Pinch lost: arm the pending window, exit when it expires.
        if self._state == _State.CURSOR:
            self._state = _State.CURSOR_PENDING
            self._pending_since = now_s
            self._pending_explicit = pose != Pose.UNKNOWN
            self._cursor_active = False
            return []
        if (now_s - self._pending_since) * 1000 > self.cursor_lost_ms:
            return self._leave_cursor(now_s, hand_gone=False)
        return []

    def _leave_cursor(self, now_s: float, hand_gone: bool) -> list[GestureEvent]:
        logger.info("cursor mode OFF (%s)", "hand gone" if hand_gone else "pose changed")
        self.reset()
        if not hand_gone:
            self._candidate_since = now_s
        return []

    def _start_cooldown(self, now_s: float) -> None:
        self._state = _State.COOLDOWN
        self._cooldown_until = now_s + self.cooldown_ms / 1000.0


def _dominant_direction(dx: float, dy: float) -> Direction:
    """Return the swipe direction only when one axis clearly dominates."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return Direction.NONE
    if abs(dx) > abs(dy) * 1.6:
        return Direction.RIGHT if dx > 0 else Direction.LEFT
    if abs(dy) > abs(dx) * 1.6:
        return Direction.DOWN if dy > 0 else Direction.UP
    return Direction.NONE
