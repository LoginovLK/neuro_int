"""Official MediaPipe GestureRecognizer backend (tasks API).

The hand-tracking industry standard: Google's own pipeline (palm detect ->
landmark track, all preprocessing matched to training exactly). Everything
manually ported before (ONNX palm + sparse landmark, DWPose wholebody)
either slid or hallucinated; this is the reference implementation Lev's
original v1 used - the quality he remembers.

The recognizer is the landmarker PLUS a trained gesture classifier head:
each frame gets Google's canned gesture (Closed_Fist / Victory /
Open_Palm / ...) with a score. The canned classes only CONFIRM poses
(tilt-stable evidence); clicks and drags are distance-based on the raw
landmarks (thumb-to-fingertip), which tilt cannot fake.

Carried over from the rest of the stack: One Euro smoothing, shape lock
(frozen skeleton re-anchored by a filtered rigid transform), static
texture-latch lockout, YuNet face gate.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from math import atan2, cos, hypot, sin
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from neuroint_cursor.pose import active_use, classify, palm_length
from neuroint_cursor.smoothing import PointFilter2D

logger = logging.getLogger(__name__)

SHAPE_REFIT = 0.55
# fingertip divergence alone forces an instant re-freeze: when the gesture
# changes, only the tips move (the palm stays), the all-point median never
# reached the threshold and the old pose stuck until the hand left the
# frame (Lev's report)
TIPS_REFIT = 0.40
TIPS = (8, 12, 16, 20)
# static texture-latch (hair/chair stick): the VIDEO tracker latches onto
# textures and redraws a hand forever; IMAGE-mode on the same frame finds
# nothing. A hand frozen in place for this long while the cursor is NOT
# engaged is locked out.
STATIC_LATCH_SECONDS = 4.0
# coast: when tracking drops (fast swipe, blur) keep the last skeleton
# FROZEN in place instead of blinking off and re-drawing at the new spot.
# Absolute deadline: _last_fresh only moves on REAL detections.
COAST_SECONDS = 0.35


class MediaPipeHands:
    """Same detect() interface as OnnxHands: [(21 normalized pts, handed, score)]."""

    def __init__(
        self,
        model_path: str | Path,
        face_model: str | Path | None = None,
        clahe: bool = False,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.GestureRecognizer.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._srgb = mp.ImageFormat.SRGB
        self._last_ts = 0
        # official gesture class of the SELECTED hand, refreshed per frame:
        # (category_name | None, score) - "None" (no canned match) is
        # normalized to (None, 0.0). The canned classes (Closed_Fist,
        # Victory, ...) only CONFIRM poses; clicks are distance-based.
        self.last_gesture: tuple[str | None, float] = (None, 0.0)
        # RAW (un-shaped, un-smoothed) normalized landmarks of the selected
        # hand: gesture distances (thumb-to-fingertip pinch) must read the
        # live cloud - a frozen shape would freeze the pinch distance too.
        self.last_raw: list[tuple[float, float]] | None = None
        logger.info("mediapipe gesture recognizer ready")

        self._face_det = None
        if face_model is not None and Path(face_model).exists():
            try:
                self._face_det = cv2.FaceDetectorYN.create(
                    str(face_model), "", (640, 360), score_threshold=0.6
                )
            except Exception as exc:  # noqa: BLE001 - gate is optional
                logger.warning("face gate unavailable: %s", exc)
        self._face_cache: tuple[float, list[tuple[float, float, float, float]]] = (
            0.0,
            [],
        )

        self._filters: list[PointFilter2D] | None = None
        self._shape: tuple[list[tuple[float, float]], float] | None = None
        self._anchor_f: PointFilter2D | None = None
        self._ang_vec: tuple[float, float] | None = None
        self._hand_f = 0.0
        # wrist of the last selected hand (normalized): sticky selection
        # keeps ONE hand while the cursor is engaged instead of flipping
        # between two candidates on score crossings
        self._sel_wrist: tuple[float, float] | None = None
        self._last_diag = 0.0
        self._coast: tuple[list[tuple[float, float]], str, float] | None = None
        # timestamp of the last REAL detection (coast does not update it):
        # control logic treats results older than ~0.15s as "no hand", and
        # the coast window ends COAST_SECONDS after it
        self._last_fresh = 0.0
        # static hallucination lock (Lev's hair stick): MediaPipe's VIDEO
        # tracker latches onto textures (hair, chair) and re-draws a hand
        # forever; IMAGE-mode on the same frame finds nothing. A candidate
        # that stays put (box IoU / center median) for STATIC_LATCH_SECONDS
        # while the cursor is NOT engaged is locked out; deliberate use at
        # that spot (open hand / pinch) re-opens it.
        self._shist: deque[tuple[float, float, float, float, float]] = deque()
        self._slock: tuple[float, float, float, float] | None = None
        # tracker-latch watchdog: the VIDEO landmarker latches onto
        # textures and redraws a hand forever; an independent IMAGE-mode
        # landmarker re-checks the same frame every second, and after two
        # consecutive "video sees a hand / image sees none" the video
        # tracker is REBUILT (a track reset it cannot self-recover from)
        self._vopts = options
        self._iopts = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
        self._img_lm = vision.GestureRecognizer.create_from_options(self._iopts)
        # The watchdog lives on its OWN daemon thread: a full IMAGE-mode
        # inference inside the 30fps loop stalled one frame every second
        # (Lev: "детекция каждую секунду подтормаживает"). This thread
        # only ever touches _img_lm; the main loop only touches
        # _landmarker — no shared landmarker, no locks needed.
        self._watch: tuple[np.ndarray, bool] | None = None
        self._stale_strikes = 0
        self._stale_trigger = False
        threading.Thread(
            target=self._watchdog_loop, name="mp-watchdog", daemon=True
        ).start()

    # ---------------- tracker-latch watchdog (own thread) ----------------

    def _watchdog_loop(self) -> None:
        """1 Hz, off the hot loop: re-check the latest frame with an
        independent IMAGE-mode landmarker. The VIDEO tracker latches
        onto textures and redraws a hand forever; two consecutive
        "video sees a hand / image sees none" verdicts queue a rebuild,
        which the main loop performs (it owns _landmarker). When the
        VIDEO result has no hand there is nothing to latch — the
        expensive re-check is skipped entirely."""
        while True:
            time.sleep(1.0)
            if self._stale_trigger:
                continue  # a rebuild is already queued, wait for it
            watch = self._watch
            if watch is None or not watch[1]:
                self._stale_strikes = 0
                continue
            frame = watch[0]
            try:
                mp_image = self._mp_image_cls(image_format=self._srgb, data=frame)
                img_res = self._img_lm.recognize(mp_image)
            except Exception as exc:  # noqa: BLE001 - watchdog must survive
                logger.debug("watchdog detect failed: %s", exc)
                continue
            if img_res.hand_landmarks:
                self._stale_strikes = 0
            else:
                self._stale_strikes += 1
                if self._stale_strikes >= 2:
                    self._stale_trigger = True
                    logger.warning("tracker latched on texture: rebuild queued")

    # ---------------- helpers (carried over) ----------------

    def _face_boxes(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        """YuNet, cached for 150ms: a face does not jump frame to frame,
        and a per-frame CPU detector steals ~10ms from the hot loop."""
        if self._face_det is None:
            return []
        now_s = time.monotonic()
        if now_s - self._face_cache[0] < 0.15:
            return self._face_cache[1]
        image_h, image_w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (640, 360))
        self._face_det.setInputSize((640, 360))
        _, faces = self._face_det.detect(small)
        if faces is None:
            out: list[tuple[float, float, float, float]] = []
        else:
            sx, sy = image_w / 640.0, image_h / 360.0
            out = [(f[0] * sx, f[1] * sy, f[2] * sx, f[3] * sy) for f in faces]
        self._face_cache = (now_s, out)
        return out

    def _shape_apply(
        self, raw: list[tuple[float, float]], now_s: float,
        image_w: int, image_h: int, force_refit: bool = False,
    ) -> list[tuple[float, float]]:
        """Frozen skeleton, rigidly re-anchored by FILTERED parameters."""
        px = [(x * image_w, y * image_h) for x, y in raw]
        hand = hypot(px[9][0] - px[0][0], px[9][1] - px[0][1])
        if hand < 8.0:
            return raw
        ang = atan2(px[9][1] - px[0][1], px[9][0] - px[0][0])
        if force_refit or self._shape is None:
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
        result = self._landmarker.recognize_for_video(mp_image, ts_ms)

        # watchdog bookkeeping (O(1)): publish the fresh frame and the
        # "video saw a hand" verdict as ONE tuple so the background thread
        # always reads a matching pair. `rgb` is a unique array per frame
        # and never mutated afterwards.
        self._watch = (rgb, bool(result.hand_landmarks))

        # the watchdog thread flagged a texture latch (2 strikes): rebuild
        # the VIDEO tracker here on the main thread — it owns _landmarker.
        # Rare (latch events only), the per-second re-check is NOT here.
        if self._stale_trigger:
            self._stale_trigger = False
            self._stale_strikes = 0
            logger.warning("tracker latch confirmed: rebuilding VIDEO tracker")
            try:
                self._landmarker.close()
            except Exception as exc:  # noqa: BLE001 - rebuild must survive
                logger.debug("landmarker close failed: %s", exc)
            self._landmarker = vision.GestureRecognizer.create_from_options(
                self._vopts
            )
            self._coast = None
            self._shist.clear()
            self._slock = None
            self._shape = None
            self._anchor_f = None
            self._ang_vec = None
            self._filters = None
            self._sel_wrist = None
            self.last_raw = None
            return []

        faces = self._face_boxes(frame_bgr)
        diag_due = now_s - self._last_diag >= 5.0

        candidates: list[tuple[float, str, list[tuple[float, float]]]] = []
        cand_gest: list[tuple[str | None, float]] = []
        n_raw = len(result.hand_landmarks)
        min_dim = min(image_w, image_h)
        for i, (lm_list, handedness) in enumerate(
            zip(result.hand_landmarks, result.handedness, strict=True)
        ):
            pts = [(lm.x, lm.y) for lm in lm_list]
            xs = [p[0] * image_w for p in pts]
            ys = [p[1] * image_h for p in pts]
            bcx, bcy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            # face gate: a hand candidate sitting inside a detected face (or in
            # the hair cap above it) is the classic MediaPipe false positive
            on_face = any(
                fx - 0.1 * fw <= bcx <= fx + 1.1 * fw
                and fy - 0.6 * fh <= bcy <= fy + 1.1 * fh
                and (max(xs) - min(xs)) <= 1.3 * fw
                for fx, fy, fw, fh in faces
            )
            if on_face:
                continue
            # static latch: the frozen hand occupying the locked spot is
            # invisible; deliberate use at that spot (open hand or pinch)
            # re-opens the lock
            if self._slock is not None:
                lx0, ly0, lx1, ly1 = self._slock
                ioux = min(max(xs), lx1) - max(min(xs), lx0)
                iouy = min(max(ys), ly1) - max(min(ys), ly0)
                inter = ioux * iouy if ioux > 0 and iouy > 0 else 0.0
                a1 = (max(xs) - min(xs)) * (max(ys) - min(ys))
                a2 = (lx1 - lx0) * (ly1 - ly0)
                iou = inter / max(a1 + a2 - inter, 1e-6)
                if iou >= 0.25:
                    if active_use(pts):
                        self._slock = None  # deliberate use retook the spot
                    else:
                        continue  # the latch itself: invisible
            cat = handedness[0]
            candidates.append((float(cat.score), cat.category_name, pts))
            # official canned gesture of THIS hand ("None" -> no match)
            gcat = None
            if i < len(result.gestures) and result.gestures[i]:
                gcat = result.gestures[i][0]
            if gcat is not None and gcat.category_name != "None":
                cand_gest.append((gcat.category_name, float(gcat.score)))
            else:
                cand_gest.append((None, 0.0))
        if diag_due:
            self._last_diag = now_s
            logger.info(
                "diag: mp hands %d -> kept %d  faces %d",
                n_raw,
                len(candidates),
                len(faces),
            )
        if not candidates:
            # coast: no blink-off — the last skeleton stays FROZEN in
            # place, at most COAST_SECONDS past the last real detection
            if (
                self._coast is not None
                and now_s - self._last_fresh <= COAST_SECONDS
            ):
                pts, handed, score = self._coast
                return [(pts, handed, score)]
            self._shape = None
            self._anchor_f = None
            self._ang_vec = None
            self._filters = None
            self._sel_wrist = None
            self.last_raw = None
            return []
        # sticky selection while engaged: keep the hand whose wrist is near
        # the previously selected one (within ~1.5 palm lengths) instead of
        # flipping between two candidates when scores cross
        best_i = 0
        if prefer_pinch and self._sel_wrist is not None:
            near = [
                i
                for i, (_s, _h, pts) in enumerate(candidates)
                if palm_length(pts) > 1e-6
                and hypot(
                    pts[0][0] - self._sel_wrist[0], pts[0][1] - self._sel_wrist[1]
                )
                <= 1.5 * palm_length(pts)
            ]
            if near:
                best_i = max(near, key=lambda i: candidates[i][0])
            else:
                best_i = max(range(len(candidates)), key=lambda i: candidates[i][0])
        else:
            best_i = max(range(len(candidates)), key=lambda i: candidates[i][0])
        score, handed, pts21 = candidates[best_i]
        self._sel_wrist = pts21[0]
        self.last_gesture = cand_gest[best_i]
        self.last_raw = pts21

        # static latch accumulator: a hand frozen in place (center hovers
        # near the window median) without ever pinching is a texture latch
        pts_px = [(p[0] * image_w, p[1] * image_h) for p in pts21]
        bb = (
            min(p[0] for p in pts_px),
            min(p[1] for p in pts_px),
            max(p[0] for p in pts_px),
            max(p[1] for p in pts_px),
        )
        if prefer_pinch:
            self._shist.clear()
        else:
            self._shist.append((now_s, *bb))
            horizon = now_s - STATIC_LATCH_SECONDS
            while self._shist and self._shist[0][0] < horizon:
                self._shist.popleft()
            if len(self._shist) >= 25:
                cxs = [(c[1] + c[3]) / 2 for c in self._shist]
                cys = [(c[2] + c[4]) / 2 for c in self._shist]
                mcx, mcy = median(cxs), median(cys)
                if hypot(cxs[-1] - mcx, cys[-1] - mcy) <= 0.10 * min_dim:
                    lx0, ly0, lx1, ly1 = self._shist[-1][1:]
                    pad = 0.25 * max(lx1 - lx0, ly1 - ly0)
                    self._slock = (lx0 - pad, ly0 - pad, lx1 + pad, ly1 + pad)
                    self._shist.clear()
                    self._coast = None
                    logger.warning("static texture latch locked out")
                    return []

        shaped = self._shape_apply(pts21, now_s, image_w, image_h)
        # gesture-change fix (Lev): if the RECOGNIZED gesture differs
        # between the raw cloud and the frozen shape, the pose actually
        # changed - re-freeze from the live cloud this same frame. The
        # tip-divergence threshold alone missed partial switches (that
        # is why gestures used to switch only through an open palm).
        if self._shape is not None and classify(pts21) != classify(shaped):
            shaped = self._shape_apply(
                pts21, now_s, image_w, image_h, force_refit=True
            )
        points = self._smooth(shaped, now_s)

        self._coast = (points, handed, score)
        self._last_fresh = now_s
        return [(points, handed, score)]
