"""Official MediaPipe HandLandmarker backend (tasks API).

The hand-tracking industry standard: Google's own pipeline (palm detect ->
landmark track, all preprocessing matched to training exactly). Everything
manually ported before (ONNX palm + sparse landmark, DWPose wholebody)
either slid or hallucinated; this is the reference implementation Lev's
original v1 used - the quality he remembers.

Carried over from the rest of the stack: pinch bundle gate, One Euro
smoothing, shape lock (frozen skeleton re-anchored by a filtered rigid
transform), still-pinch sticker lockout, YuNet face gate.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from math import atan2, cos, hypot, sin
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from neuroint_cursor.pose import is_pinch, pinch_center
from neuroint_cursor.smoothing import PointFilter2D

logger = logging.getLogger(__name__)

STILL_PINCH_SECONDS = 5.0
STILL_TIP_SPREAD = 0.02
LOCK_UNLOCK_PX = 0.06
SHAPE_REFIT = 0.55
# fingertip divergence alone forces an instant re-freeze: when the gesture
# changes, only the tips move (the palm stays), the all-point median never
# reached the threshold and the old pose stuck until the hand left the
# frame (Lev's report)
TIPS_REFIT = 0.55
TIPS = (8, 12, 16, 20)
# coast: when tracking drops (fast swipe, blur) keep the last skeleton and
# carry it along the last bundle velocity instead of blinking off and
# re-drawing at the new spot
COAST_SECONDS = 0.35
COAST_VEL_CAP = 1500.0  # px/s


class MediaPipeHands:
    """Same detect() interface as OnnxHands: [(21 normalized pts, handed, score)]."""

    def __init__(
        self,
        model_path: str | Path,
        face_model: str | Path | None = None,
        clahe: bool = False,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._srgb = mp.ImageFormat.SRGB
        self._last_ts = 0
        logger.info("mediapipe hand landmarker ready")

        self._face_det = None
        if face_model is not None and Path(face_model).exists():
            try:
                self._face_det = cv2.FaceDetectorYN.create(
                    str(face_model), "", (640, 360), score_threshold=0.6
                )
            except Exception as exc:  # noqa: BLE001 - gate is optional
                logger.warning("face gate unavailable: %s", exc)

        self._filters: list[PointFilter2D] | None = None
        self._shape: tuple[list[tuple[float, float]], float] | None = None
        self._anchor_f: PointFilter2D | None = None
        self._ang_vec: tuple[float, float] | None = None
        self._hand_f = 0.0
        self._tip_hist: deque[tuple[float, float, float]] = deque()
        self._lock: tuple[float, float, float, float] | None = None
        self._lock_seen = 0.0
        self._last_diag = 0.0
        self._coast: tuple[list[tuple[float, float]], str, float] | None = None
        self._coast_t = 0.0
        self._bundle_prev: tuple[float, float] | None = None
        self._bundle_vel: tuple[float, float] = (0.0, 0.0)
        # timestamp of the last REAL detection (coast does not update it):
        # control logic treats results older than ~0.1s as "no hand"
        self._last_fresh = 0.0

    # ---------------- helpers (carried over) ----------------

    def _face_boxes(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        if self._face_det is None:
            return []
        image_h, image_w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (640, 360))
        self._face_det.setInputSize((640, 360))
        _, faces = self._face_det.detect(small)
        if faces is None:
            return []
        sx, sy = image_w / 640.0, image_h / 360.0
        return [(f[0] * sx, f[1] * sy, f[2] * sx, f[3] * sy) for f in faces]

    def _shape_apply(
        self, raw: list[tuple[float, float]], now_s: float,
        image_w: int, image_h: int,
    ) -> list[tuple[float, float]]:
        """Frozen skeleton, rigidly re-anchored by FILTERED parameters."""
        px = [(x * image_w, y * image_h) for x, y in raw]
        hand = hypot(px[9][0] - px[0][0], px[9][1] - px[0][1])
        if hand < 8.0:
            return raw
        ang = atan2(px[9][1] - px[0][1], px[9][0] - px[0][0])
        if self._shape is None:
            self._shape = (
                [((p[0] - px[0][0]) / hand, (p[1] - px[0][1]) / hand) for p in px],
                ang,
            )
            self._anchor_f = PointFilter2D(min_cutoff=1.0, beta=0.02)
            self._ang_vec = (cos(ang), sin(ang))
            self._hand_f = hand
            return raw
        offsets, ang0 = self._shape

        fx, fy = self._anchor_f.filter(px[0][0], px[0][1], now_s)
        vx, vy = self._ang_vec
        vx, vy = 0.85 * vx + 0.15 * cos(ang), 0.85 * vy + 0.15 * sin(ang)
        norm = hypot(vx, vy)
        if norm > 1e-6:
            vx, vy = vx / norm, vy / norm
        self._ang_vec = (vx, vy)
        ang_f = atan2(vy, vx)
        self._hand_f = 0.85 * self._hand_f + 0.15 * hand

        d = ang_f - ang0
        ca, sa = cos(d), sin(d)
        emitted = [
            (fx + self._hand_f * (ox * ca - oy * sa),
             fy + self._hand_f * (ox * sa + oy * ca))
            for ox, oy in offsets
        ]
        div = median(
            hypot(emitted[i][0] - px[i][0], emitted[i][1] - px[i][1])
            for i in range(21)
        ) / hand
        tips_div = median(
            hypot(emitted[i][0] - px[i][0], emitted[i][1] - px[i][1]) for i in TIPS
        ) / hand
        if tips_div > TIPS_REFIT:
            # gesture change: the tips moved while the palm stayed —
            # re-freeze hard from the live cloud
            self._shape = (
                [((p[0] - px[0][0]) / hand, (p[1] - px[0][1]) / hand) for p in px],
                ang_f,
            )
            return [(x / image_w, y / image_h) for x, y in px]
        if div > SHAPE_REFIT:
            self._shape = (
                [((p[0] - px[0][0]) / hand, (p[1] - px[0][1]) / hand) for p in px],
                ang_f,
            )
        return [(ex / image_w, ey / image_h) for ex, ey in emitted]

    def _smooth(
        self, raw_points: list[tuple[float, float]], now_s: float
    ) -> list[tuple[float, float]]:
        if self._filters is None:
            self._filters = [PointFilter2D(min_cutoff=1.0, beta=0.02) for _ in range(21)]
        return [
            (float(sx), float(sy))
            for sx, sy in (
                self._filters[i].filter(x, y, now_s)
                for i, (x, y) in enumerate(raw_points)
            )
        ]

    def _sticker_check(
        self,
        points: list[tuple[float, float]],
        pinch_now: bool,
        now_s: float,
        image_w: int,
        image_h: int,
    ) -> bool:
        min_dim = min(image_w, image_h)
        bfx, bfy = pinch_center(points)
        cx, cy = bfx * image_w, bfy * image_h
        if self._lock is not None:
            moved = hypot(cx - self._lock[0], cy - self._lock[1])
            if not pinch_now or moved > LOCK_UNLOCK_PX * min_dim:
                self._lock = None
            else:
                if now_s - self._lock_seen > 0.5:
                    self._lock_seen = now_s
                return True
        if pinch_now:
            self._tip_hist.append((now_s, cx, cy))
            horizon = now_s - STILL_PINCH_SECONDS
            while self._tip_hist and self._tip_hist[0][0] < horizon:
                self._tip_hist.popleft()
            if (
                len(self._tip_hist) >= 30
                and self._tip_hist[-1][0] - self._tip_hist[0][0]
                >= STILL_PINCH_SECONDS * 0.8
            ):
                xs = [x for _, x, _y in self._tip_hist]
                ys = [y for _, _x, y in self._tip_hist]
                if hypot(max(xs) - min(xs), max(ys) - min(ys)) <= STILL_TIP_SPREAD * min_dim:
                    self._lock = (cx, cy, 0.0, 0.0)
                    self._lock_seen = now_s
                    self._tip_hist.clear()
                    self._coast = None  # a locked sticker must not coast
                    logger.warning("still-pinch sticker locked out")
                    return True
        else:
            self._tip_hist.clear()
        return False

    # ---------------- public ----------------

    def detect(
        self, frame_bgr: np.ndarray, prefer_pinch: bool = False
    ) -> list[tuple[list[tuple[float, float]], str, float]]:
        """Full frame BGR -> [(21 normalized points, handedness, score)]."""
        image_h, image_w = frame_bgr.shape[:2]
        now_s = time.monotonic()
        ts_ms = max(int(now_s * 1000), self._last_ts + 1)
        self._last_ts = ts_ms

        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        mp_image = self._mp_image_cls(image_format=self._srgb, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        faces = self._face_boxes(frame_bgr)
        diag_due = now_s - self._last_diag >= 5.0

        candidates: list[tuple[float, str, list[tuple[float, float]]]] = []
        n_raw = len(result.hand_landmarks)
        for lm_list, handedness in zip(
            result.hand_landmarks, result.handedness, strict=True
        ):
            pts = [(lm.x, lm.y) for lm in lm_list]
            xs = [p[0] * image_w for p in pts]
            ys = [p[1] * image_h for p in pts]
            bcx, bcy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            # face gate: a hand candidate sitting inside a detected face is
            # the classic MediaPipe face false positive
            on_face = any(
                fx - 0.1 * fw <= bcx <= fx + 1.1 * fw
                and fy - 0.1 * fh <= bcy <= fy + 1.1 * fh
                and (max(xs) - min(xs)) <= 1.3 * fw
                for fx, fy, fw, fh in faces
            )
            if on_face:
                continue
            cat = handedness[0]
            candidates.append((float(cat.score), cat.category_name, pts))
        if diag_due:
            self._last_diag = now_s
            logger.info(
                "diag: mp hands %d -> kept %d  faces %d",
                n_raw,
                len(candidates),
                len(faces),
            )
        if not candidates:
            # coast: no blink-off — carry the last skeleton along the last
            # bundle velocity for a short grace period
            if (
                self._coast is not None
                and self._lock is None
                and now_s - self._coast_t <= COAST_SECONDS
            ):
                dt = max(now_s - self._coast_t, 1 / 120)
                dx = max(
                    min(self._bundle_vel[0] * dt, COAST_VEL_CAP * dt),
                    -COAST_VEL_CAP * dt,
                )
                dy = max(
                    min(self._bundle_vel[1] * dt, COAST_VEL_CAP * dt),
                    -COAST_VEL_CAP * dt,
                )
                pts, handed, score = self._coast
                pts = [(p[0] + dx / image_w, p[1] + dy / image_h) for p in pts]
                self._coast = (pts, handed, score)
                self._coast_t = now_s
                return [(pts, handed, score)]
            self._shape = None
            self._anchor_f = None
            self._ang_vec = None
            self._filters = None
            self._bundle_prev = None
            self._bundle_vel = (0.0, 0.0)
            return []
        if prefer_pinch:
            pinched = [c for c in candidates if is_pinch(c[2])]
            if pinched:
                candidates = pinched
        score, handed, pts21 = max(candidates, key=lambda c: c[0])

        shaped = self._shape_apply(pts21, now_s, image_w, image_h)
        points = self._smooth(shaped, now_s)

        # bundle velocity for coasting
        bfx, bfy = pinch_center(points)
        bx, by = bfx * image_w, bfy * image_h
        if self._bundle_prev is not None:
            dt = max(now_s - self._coast_t, 1 / 120)
            nvx = (bx - self._bundle_prev[0]) / dt
            nvy = (by - self._bundle_prev[1]) / dt
            nvx = max(min(nvx, COAST_VEL_CAP), -COAST_VEL_CAP)
            nvy = max(min(nvy, COAST_VEL_CAP), -COAST_VEL_CAP)
            self._bundle_vel = (
                0.6 * self._bundle_vel[0] + 0.4 * nvx,
                0.6 * self._bundle_vel[1] + 0.4 * nvy,
            )
        self._bundle_prev = (bx, by)

        pinch_now = is_pinch(points)
        if self._sticker_check(points, pinch_now, now_s, image_w, image_h):
            self._coast = None
            return []
        self._coast = (points, handed, score)
        self._coast_t = now_s
        self._last_fresh = now_s
        return [(points, handed, score)]
