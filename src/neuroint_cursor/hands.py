"""ONNX hand detection: the MediaPipe two-stage pipeline on the GPU.

Ports the PINTO0309/hand-gesture-recognition-using-onnx pipeline:
  1. palm_detection_full_inf_post_192x192.onnx — palm detector with NMS
     built into the ONNX graph
  2. hand_landmark_sparse_Nx3x224x224.onnx — 21 landmarks per hand

Runs through onnxruntime DirectML on the RTX GPU. Detection is
track-priority (this is what keeps the cursor from jumping):

- while a track is alive and valid, the tracked ROI (previous landmarks
  + velocity extrapolation) is THE source — smooth and continuous;
- the palm detector runs every frame as a scout: near candidates are
  duplicates of the tracked hand and are ignored; a FAR candidate that
  scores much better is a different hand and takes over;
- when the tracked crop misses (motion blur, fast swipe), a palm
  candidate near the last position bridges the gap WITHOUT resetting the
  filters — no snap;
- when there is no track at all, the best palm candidate initializes one
  (filter reset — unavoidable, position really changed);
- palm threshold adapts: 0.4 with a live track, 0.3 without (far/small
  hands); phantoms this lets in are handled by static-object quarantine.

Crop->frame decode is the exact inverse of the rotated-crop transform
(rotation included; verified empirically with a synthetic marker) — a
translation-only decode is off by sin(angle)*distance on tilted hands.
CLAHE runs on the palm frame and every crop (dark-room contrast).
Tracked ROI size/center are limited per frame AND over a 1.5s window, so
one garbage frame cannot blow the crop into a feedback spiral.

Decode fixes from the old pipeline preserved verbatim:
- rr_cx is already normalized to the frame (double-normalization bug)
- rr_cy un-pad needs the /scale step (hand drifting 200px bug)
- BORDER_REFLECT_101 padding in rotate_and_crop (edge-of-frame bug)
- int() casts on the float32 rect before numpy slicing
- run(...)[0] unpack (not [0][0]) for the [N,8] palm output
"""

from __future__ import annotations

import logging
import time
from collections import deque
from math import atan2, cos, hypot, pi, sin
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from neuroint_cursor.pose import is_pointing, palm_ratio, palm_sane
from neuroint_cursor.smoothing import PointFilter2D
from neuroint_cursor.tracker import HandTemplateTracker
from neuroint_cursor.tracking import (
    TRACK_EMPTY_PALM_LIMIT,
    TRACK_MISS_LIMIT,
    TRACK_SCORE_MIN,
    normalize_radians,
    rect_from_landmarks,
)

logger = logging.getLogger(__name__)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# A palm candidate FAR from the track (center distance above this factor
# of the tracked ROI size) is a different object, not the tracked hand.
NEAR_FACTOR = 1.5
# ...and it only takes over when it beats the track by this margin
# (a different hand must be clearly better, not noise-level better).
SWITCH_MARGIN = 0.15
# Bonus for a pointing candidate: a pointing hand must outscore a
# non-pointing anchor (face at frame center) even at equal raw scores.
POINTING_BONUS = 0.15
# Sustained non-pointing stretch (frames) before a NEAR candidate may
# take the track from a non-pointing holder: the hand in use flickers
# out of pointing for a frame now and then, and a per-frame takeover
# would switch sources and jerk the cursor.
NP_RUN_MIN = 8
# Velocity extrapolation: conservative on purpose — an EMA velocity lags
# behind direction changes, and a confident wrong prediction oscillates
# the crop around the hand (the "glitchy jerk" on sharp direction flips).
VEL_MAX_PX_S = 3000.0
VEL_PREDICT_MAX_S = 0.08
VEL_PREDICT_FACTOR = 0.6
VEL_EMA_NEW = 0.7         # weight of the fresh velocity sample

# Static-object rejection: shelf junk / furniture edges get detected as a
# palm and then tracked forever. A real hand always micro-sways, and a
# hand HELD deliberately still is in pointing pose (in use). So: an
# object whose ROI center has not moved for STATIC_SECONDS while never
# pointing is not a hand — drop it and quarantine that spot for
# QUARANTINE_SECONDS (palm candidates there are ignored).
STATIC_SECONDS = 2.0
STATIC_SPREAD = 0.10      # fraction of min(frame_w, frame_h)
QUARANTINE_SECONDS = 8.0
QUARANTINE_RADIUS = 0.08  # fraction of min(frame_w, frame_h)

# Tracked-ROI size, bounded over a sliding window: hallucinated landmarks
# inflate the bbox a few percent per frame, which compounds. A real hand
# changes size slower than this over 1.5s (fast approach is carried by
# the palm path, whose crops are unbounded).
SIZE_WINDOW_SECONDS = 1.5
SIZE_GROW_MAX = 1.3       # vs the largest size seen in the window
SIZE_SHRINK_MIN = 0.8     # vs the smallest size seen in the window
SIZE_ABS_MAX = 0.5        # fraction of min(frame_w, frame_h)

# Template tracking (Lev's architecture): once the hand points, its 21
# landmarks are FROZEN as a shape and only translated per frame by phase
# correlation (~0.5ms, no CNN, zero landmark jitter). The landmark model
# re-fits on triggers only: every REFIT_EVERY frames (drift) or right
# away when the correlation drops (pose change / blur / occlusion).
# Pose acquisition stays on the full pipeline so engage is instant.
TEMPLATE_START_SCORE = 0.55
TEMPLATE_MIN_RESPONSE = 0.12
REFIT_EVERY = 8

# Tilted hands: the palm box angle can be tens of degrees off, and the
# first landmark pass then hallucinates a SQUASHED hand (palm_ratio far
# outside the hand band) while still scoring 0.9 — the emitted direction
# is useless for correction (it points "up" regardless). On a sanity
# failure the rect is re-inferred with rotated copies and the best sane
# result wins (measured live: -17deg -> ratio 0.55 garbage, +28deg -> 0.98).
ROTATION_RETRIES = (25.0, -25.0, 50.0, -50.0)

# Face false positives: the palm model fires ALL OVER a face — several
# conflicting boxes within one box-size radius (4 measured on Lev's face
# vs 1 on a real hand). A crowded cluster is not a hand.
CLUSTER_REJECT = 3
CLUSTER_RADIUS_FACTOR = 1.0


def _keep_aspect_resize_and_pad(
    image: np.ndarray, resize_width: int, resize_height: int
) -> tuple[np.ndarray, np.ndarray]:
    image_height, image_width = image.shape[:2]
    padded = np.zeros((resize_height, resize_width, 3), np.uint8)
    asw = resize_width / image_width
    ash = resize_height / image_height
    if asw < ash:
        sizeas = (int(image_width * asw), int(image_height * asw))
    else:
        sizeas = (int(image_width * ash), int(image_height * ash))
    resized = cv2.resize(image, dsize=sizeas)
    start_h = int(resize_height / 2 - sizeas[1] / 2)
    end_h = int(resize_height / 2 + sizeas[1] / 2)
    start_w = int(resize_width / 2 - sizeas[0] / 2)
    end_w = int(resize_width / 2 + sizeas[0] / 2)
    padded[start_h:end_h, start_w:end_w, :] = resized.copy()
    return padded, resized


def _rotate_and_crop(image: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """Crop a rotated rectangle (mirror padding when it leaves the frame)."""
    # rect comes in as float32: everything used for slicing must be int.
    cx, cy, rect_w, rect_h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    angle = float(rect[4])
    # Mirror-reflection padding instead of black: a hand at the frame edge
    # gets a seamless continuation, so the landmark model keeps its score.
    pad_x = (image.shape[1] // 2) + rect_w
    pad_y = (image.shape[0] // 2) + rect_h
    padded = cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101
    )
    cx += pad_x
    cy += pad_y

    rect_tuple = ((cx, cy), (rect_w, rect_h), angle)
    box = cv2.boxPoints(rect_tuple)
    x_min, y_min = int(np.min(box[:, 0])), int(np.min(box[:, 1]))
    x_max, y_max = int(np.max(box[:, 0])), int(np.max(box[:, 1]))
    bx_cx, bx_cy = int((x_min + x_max) // 2), int((y_min + y_max) // 2)
    bx_w, bx_h = x_max - x_min, y_max - y_min

    upright = padded[
        bx_cy - bx_h // 2 : bx_cy + (bx_h - bx_h // 2),
        bx_cx - bx_w // 2 : bx_cx + (bx_w - bx_w // 2),
    ]
    height, width = upright.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, int(angle), 1)
    abs_cos, abs_sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    matrix[0, 2] += bound_w / 2 - center[0]
    matrix[1, 2] += bound_h / 2 - center[1]
    rotated = cv2.warpAffine(upright, matrix, (bound_w, bound_h))

    crop_cx, crop_cy = bound_w // 2, bound_h // 2
    return rotated[
        crop_cy - rect_h // 2 : crop_cy + (rect_h - rect_h // 2),
        crop_cx - rect_w // 2 : crop_cx + (rect_w - rect_w // 2),
    ]


class OnnxHands:
    """Two-stage hand pipeline (palm detect -> landmarks) on DirectML."""

    def __init__(
        self,
        palm_model: str | Path,
        landmark_model: str | Path,
        score_threshold: float = 0.4,
        far_palm_threshold: float = 0.3,
        class_score_threshold: float = 0.25,
        tracking: bool = True,
        clahe: bool = True,
        template: bool = True,
    ) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self._palm = ort.InferenceSession(str(palm_model), options, providers=providers)
        self._landmark = ort.InferenceSession(str(landmark_model), options, providers=providers)
        logger.info("onnx hands providers: %s", self._landmark.get_providers())

        self._palm_input = self._palm.get_inputs()[0].name
        self._palm_out = [o.name for o in self._palm.get_outputs()]
        self._lm_input = self._landmark.get_inputs()[0].name
        self._lm_out = [o.name for o in self._landmark.get_outputs()]
        palm_shape = self._palm.get_inputs()[0].shape
        self._palm_hw = (int(palm_shape[2]), int(palm_shape[3]))
        lm_shape = self._landmark.get_inputs()[0].shape
        self._lm_hw = (int(lm_shape[2]), int(lm_shape[3]))

        self._score_threshold = score_threshold
        self._far_palm_threshold = far_palm_threshold
        self._class_score_threshold = class_score_threshold
        self._tracking = tracking
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

        self._filters: list[PointFilter2D] | None = None
        self._prev_points_norm: list[tuple[float, float]] | None = None
        self._prev_dir: tuple[float, float] | None = None
        self._prev_rect: list[float] | None = None
        self._last_source: str | None = None
        self._miss = 0
        self._empty_palms = 0
        self._hist: deque[tuple[float, float, float]] = deque()
        self._size_hist: deque[tuple[float, float]] = deque()
        self._excl_until = 0.0
        self._excl_at: tuple[float, float] = (0.0, 0.0)
        self._vel: tuple[float, float] | None = None
        self._last_center: tuple[float, float] | None = None
        self._last_t: float | None = None
        self._np_run = 0
        self._template = template
        self._tpl: HandTemplateTracker | None = None

    def _drop_track(self) -> None:
        self._prev_points_norm = None
        self._prev_dir = None
        self._prev_rect = None
        self._vel = None
        self._last_center = None
        self._last_t = None
        self._np_run = 0
        self._miss = 0
        self._empty_palms = 0
        self._tpl = None
        self.reset_smoothing()

    def reset_smoothing(self) -> None:
        """Drop filter history (call when the tracked hand changes)."""
        self._filters = None

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel (proven v1 quality step; cheap on crops)."""
        if self._clahe is None:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        light, a_ch, b_ch = cv2.split(lab)
        light = self._clahe.apply(light)
        return cv2.cvtColor(cv2.merge((light, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    def _palm_rects(self, frame_bgr: np.ndarray, thr: float) -> list[list[float]]:
        """Stage 1: palm boxes -> landmark-crop rects (frame pixels)."""
        image_h, image_w = frame_bgr.shape[:2]
        palm_hw_w, palm_hw_h = self._palm_hw[1], self._palm_hw[0]
        enhanced = self._enhance(frame_bgr) if self._clahe is not None else frame_bgr
        padded, _ = _keep_aspect_resize_and_pad(enhanced, palm_hw_w, palm_hw_h)
        blob = padded / 255.0
        blob = blob[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        boxes = self._palm.run(self._palm_out, {self._palm_input: blob})[0]

        rects: list[list[float]] = []
        for box in boxes:
            score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = box
            if score <= thr or box_size <= 0:
                continue
            kp02_y = kp2_y - kp0_y
            kp02_x = kp2_x - kp0_x
            rr_size = 2.9 * box_size
            rotation = normalize_radians(0.5 * pi - atan2(-kp02_y, kp02_x))
            rr_cx = box_x + 0.5 * box_size * sin(rotation)
            rr_cy = box_y - 0.5 * box_size * cos(rotation)
            # Un-pad Y: y_px_src = (y_norm*192 - pad)/scale, then /frame_h.
            # scale = 192 / frame_w (keep-aspect, landscape).
            scale = palm_hw_w / image_w
            pad_h_px = (palm_hw_h - image_h * scale) / 2.0
            rr_cy = (rr_cy * palm_hw_h - pad_h_px) / scale / image_h
            # rr_cx/rr_cy are already normalized to the frame; rr_size is
            # normalized to the 192 model input = fraction of frame width.
            rr_size_norm = rr_size

            cx = int(rr_cx * image_w)
            cy = int(rr_cy * image_h)
            width = int(rr_size_norm * image_w)  # margin already inside (2.9x)
            height = width
            degree = np.degrees(rotation)
            rects.append([cx, cy, width, height, float(degree)])
        return rects

    def _landmark_from_rect(
        self, frame_bgr: np.ndarray, rect: list[float], min_score: float | None = None
    ) -> tuple[list[tuple[float, float]], str, float] | None:
        """Stage 2 on one rect: crop -> CLAHE -> landmarks -> normalized pts.

        Returns RAW (unsmoothed) points; smoothing happens once per frame
        on the winning candidate only.
        """
        image_h, image_w = frame_bgr.shape[:2]
        crop = _rotate_and_crop(frame_bgr, np.asarray(rect, dtype=np.float32))
        if crop.size == 0:
            return None
        crop = self._enhance(crop)
        lm_padded, lm_resized = _keep_aspect_resize_and_pad(
            crop, self._lm_hw[1], self._lm_hw[0]
        )
        lm_pad_w_half = max(0, (lm_padded.shape[1] - lm_resized.shape[1]) // 2)
        lm_pad_h_half = max(0, (lm_padded.shape[0] - lm_resized.shape[0]) // 2)
        scale_w = lm_resized.shape[1] / crop.shape[1]
        scale_h = lm_resized.shape[0] / crop.shape[0]

        blob = lm_padded / 255.0
        blob = blob[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        xyz21, hand_scores, handed = self._landmark.run(self._lm_out, {self._lm_input: blob})
        lm_score = float(hand_scores[0][0])
        if lm_score < (min_score if min_score is not None else self._class_score_threshold):
            return None

        lms = xyz21[0].reshape(21, 3) / self._lm_hw[0]
        xy = np.asarray([[p[0], p[1]] for p in lms], dtype=np.float32)
        xy[:, 0] = (xy[:, 0] * self._lm_hw[1] - lm_pad_w_half) / max(scale_w, 1e-6)
        xy[:, 1] = (xy[:, 1] * self._lm_hw[0] - lm_pad_h_half) / max(scale_h, 1e-6)

        # Crop-local pixels -> frame pixels: exact inverse of the rotated
        # crop (empirically verified with a synthetic marker):
        #   frame = center + R(+angle) @ (crop_xy - rect_size/2)
        cx, cy, rect_w, rect_h = rect[0], rect[1], rect[2], rect[3]
        a = np.radians(rect[4])
        ca, sa = np.cos(a), np.sin(a)
        ux = xy[:, 0] - rect_w / 2.0
        uy = xy[:, 1] - rect_h / 2.0
        xy[:, 0] = cx + ux * ca - uy * sa
        xy[:, 1] = cy + ux * sa + uy * ca
        xy[:, 0] /= image_w
        xy[:, 1] /= image_h

        # Display-only label: frames are mirrored at capture, so the
        # model's handedness is not used for control logic anywhere.
        handedness = "Right" if float(handed[0][0]) > 0.5 else "Left"

        points = [(float(x), float(y)) for x, y in xy]
        return points, handedness, lm_score

    def _landmark_smart(
        self, frame_bgr: np.ndarray, rect: list[float], min_score: float | None = None
    ) -> tuple[list[tuple[float, float]], str, float] | None:
        """Landmark inference with rotation retries on a collapsed palm."""
        base = self._landmark_from_rect(frame_bgr, rect, min_score)
        if base is None and min_score is not None:
            base = self._landmark_from_rect(
                frame_bgr, rect, min_score=self._class_score_threshold
            )
        if base is not None and palm_sane(base[0]):
            return base

        best = base
        best_key = None
        if best is not None:
            r = palm_ratio(best[0])
            best_key = abs(r - 1.0) if r is not None else 99.0
        eff = min_score if base is not None else self._class_score_threshold
        for off in ROTATION_RETRIES:
            rect2 = [rect[0], rect[1], rect[2], rect[3], rect[4] + off]
            out = self._landmark_from_rect(frame_bgr, rect2, min_score=eff)
            if out is None:
                continue
            if palm_sane(out[0]):
                return out
            r = palm_ratio(out[0])
            key = abs(r - 1.0) if r is not None else 99.0
            if best_key is None or key < best_key:
                best, best_key = out, key
        return best

    @staticmethod
    def _spread_ok(
        points: list[tuple[float, float]],
        rect: list[float],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """The landmark cloud must actually fill its crop (scale-free).

        A collapsed blob spans a fraction of the palm box; a real hand
        fills most of it (measured 0.6-0.9 on live frames, collapsed
        templates well below 0.35).
        """
        xs = [p[0] * frame_w for p in points]
        ys = [p[1] * frame_h for p in points]
        diag = hypot(max(xs) - min(xs), max(ys) - min(ys))
        return diag >= 0.35 * rect[2]

    @staticmethod
    def _cluster_filter(rects: list[list[float]]) -> list[list[float]]:
        """Drop boxes in crowded clusters (the model fires all over a face)."""
        keep: list[list[float]] = []
        for i, r in enumerate(rects):
            neighbors = sum(
                1
                for j, q in enumerate(rects)
                if j != i
                and hypot(r[0] - q[0], r[1] - q[1]) < CLUSTER_RADIUS_FACTOR * r[2]
            )
            if neighbors + 1 >= CLUSTER_REJECT:
                continue
            keep.append(r)
        return keep

    def _clamp_size(
        self, rect: list[float], now_s: float, frame_w: int, frame_h: int
    ) -> list[float]:
        """Bound the tracked ROI size over a sliding window (anti-runaway)."""
        self._size_hist.append((now_s, float(rect[2])))
        horizon = now_s - SIZE_WINDOW_SECONDS
        while self._size_hist and self._size_hist[0][0] < horizon:
            self._size_hist.popleft()
        sizes = [s for _, s in self._size_hist]
        lo = min(sizes) * SIZE_SHRINK_MIN
        hi = min(max(max(sizes) * SIZE_GROW_MAX, 64.0), SIZE_ABS_MAX * min(frame_w, frame_h))
        rect[2] = rect[3] = int(min(max(rect[2], lo), hi))
        return rect

    def _on_crop(
        self,
        points: list[tuple[float, float]],
        rect: list[float],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """Garbage gate: the landmark cluster must sit on the tracked crop."""
        xs = [p[0] * frame_w for p in points]
        ys = [p[1] * frame_h for p in points]
        bcx, bcy = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
        dist = ((bcx - rect[0]) ** 2 + (bcy - rect[1]) ** 2) ** 0.5
        return dist <= 0.75 * rect[2]

    def _eff(self, out: tuple[list[tuple[float, float]], str, float]) -> float:
        """Effective score: pointing candidates outscore equal raw anchors."""
        return out[2] + (POINTING_BONUS if is_pointing(out[0]) else 0.0)

    def _smooth(
        self, raw_points: list[tuple[float, float]], now_s: float
    ) -> list[tuple[float, float]]:
        """Winner-level One Euro — one update per landmark per frame."""
        if self._filters is None:
            self._filters = [PointFilter2D(min_cutoff=1.0, beta=0.02) for _ in range(21)]
        return [
            (float(sx), float(sy))
            for sx, sy in (
                self._filters[i].filter(x, y, now_s)
                for i, (x, y) in enumerate(raw_points)
            )
        ]

    def _update_velocity(self, center: tuple[float, float], now_s: float) -> None:
        if self._last_center is not None and self._last_t is not None:
            dt = max(now_s - self._last_t, 1e-3)
            nvx = (center[0] - self._last_center[0]) / dt
            nvy = (center[1] - self._last_center[1]) / dt
            nvx = max(min(nvx, VEL_MAX_PX_S), -VEL_MAX_PX_S)
            nvy = max(min(nvy, VEL_MAX_PX_S), -VEL_MAX_PX_S)
            if self._vel is None:
                self._vel = (nvx, nvy)
            else:
                self._vel = (
                    (1.0 - VEL_EMA_NEW) * self._vel[0] + VEL_EMA_NEW * nvx,
                    (1.0 - VEL_EMA_NEW) * self._vel[1] + VEL_EMA_NEW * nvy,
                )
        self._last_center = center
        self._last_t = now_s

    def _static_check(
        self,
        points: list[tuple[float, float]],
        center: tuple[float, float],
        now_s: float,
        image_w: int,
        image_h: int,
    ) -> bool:
        """True when the object was just quarantined (still, never pointing)."""
        self._hist.append((now_s, center[0], center[1]))
        horizon = now_s - STATIC_SECONDS
        while self._hist and self._hist[0][0] < horizon:
            self._hist.popleft()
        if (
            len(self._hist) >= 15
            and self._hist[-1][0] - self._hist[0][0] >= STATIC_SECONDS * 0.8
            and not is_pointing(points)
        ):
            cxs = [c for _, c, _ in self._hist]
            cys = [c for _, _, c in self._hist]
            spread = hypot(max(cxs) - min(cxs), max(cys) - min(cys))
            if spread <= STATIC_SPREAD * min(image_w, image_h):
                self._excl_at = (cxs[-1] / image_w, cys[-1] / image_h)
                self._excl_until = now_s + QUARANTINE_SECONDS
                self._hist.clear()
                self._drop_track()
                logger.info(
                    "static object at (%.2f, %.2f) quarantined %.0fs",
                    self._excl_at[0],
                    self._excl_at[1],
                    QUARANTINE_SECONDS,
                )
                return True
        return False

    def detect(
        self, frame_bgr: np.ndarray, prefer_pointing: bool = False, engaged: bool = False
    ) -> list[tuple[list[tuple[float, float]], str, float]]:
        """Full frame BGR -> list of (21 normalized landmarks, handedness, score).

        prefer_pointing: the cursor is engaged — bridging a missed track
        only accepts pointing candidates (a face anchor at frame center
        must not steal the hand mid-use).
        engaged: the main loop's debounce confirmed pointing — only then
        may the template be captured (a single flicker frame must not
        freeze a garbage pose).
        """
        image_h, image_w = frame_bgr.shape[:2]
        now_s = time.monotonic()

        # --- template fast path: frozen pointing pose, translation only ---
        if self._template and self._tpl is not None and self._tpl.active:
            if self._tpl.age < REFIT_EVERY:
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                res = self._tpl.update(frame_gray)
                if res is None:
                    # correlation dropped: pose change, blur, occlusion, edge
                    self._tpl = None  # fall through to the full pipeline
                else:
                    raw_points, handed, score, center = res
                    points = self._smooth(raw_points, now_s)
                    self._prev_points_norm = points
                    if self._prev_rect is not None:
                        self._prev_rect[0] = int(round(center[0]))
                        self._prev_rect[1] = int(round(center[1]))
                    self._update_velocity(center, now_s)
                    if self._static_check(points, center, now_s, image_w, image_h):
                        return []
                    return [(points, handed, score)]
            # else: sync frame — the full pipeline below re-fits the template

        # --- tracked candidate: ROI = previous landmarks + velocity ---
        track_out: tuple[list[tuple[float, float]], str, float] | None = None
        track_rect: list[float] | None = None
        track_alive = self._tracking and self._prev_points_norm is not None
        if track_alive:
            rect, self._prev_dir = rect_from_landmarks(
                self._prev_points_norm, image_w, image_h, self._prev_dir, self._prev_rect
            )
            if rect is not None:
                rect = self._clamp_size(rect, now_s, image_w, image_h)
                if self._vel is not None and self._last_t is not None:
                    dt = min(max(now_s - self._last_t, 1e-3), VEL_PREDICT_MAX_S)
                    rect[0] += int(
                        max(
                            min(self._vel[0] * VEL_PREDICT_FACTOR * dt, 0.25 * image_w),
                            -0.25 * image_w,
                        )
                    )
                    rect[1] += int(
                        max(
                            min(self._vel[1] * VEL_PREDICT_FACTOR * dt, 0.25 * image_h),
                            -0.25 * image_h,
                        )
                    )
                out = self._landmark_smart(frame_bgr, rect, min_score=TRACK_SCORE_MIN)
                if out is not None and self._on_crop(out[0], rect, image_w, image_h):
                    track_out = out
                    track_rect = rect

        # --- palm detector as scout ---
        # With a live track: 0.4 (high bar). Without: 0.3 — far/small hands
        # the strict bar would miss; phantoms get quarantined by stillness.
        palm_thr = self._score_threshold if track_alive else self._far_palm_threshold
        quarantining = now_s < self._excl_until
        qx, qy = self._excl_at
        qrad = QUARANTINE_RADIUS * min(image_w, image_h)
        palm_rects = self._cluster_filter(self._palm_rects(frame_bgr, palm_thr))
        palm_cands: list[tuple[tuple[list[tuple[float, float]], str, float], list[float]]] = []
        for rect in palm_rects:
            # A big palm box is a close real hand — never quarantine-block it
            # (junk blobs are small; the hand at working distance is large).
            big_hand = rect[2] > 0.18 * image_w
            if quarantining and not big_hand and hypot(rect[0] - qx, rect[1] - qy) <= qrad:
                continue
            out = self._landmark_smart(frame_bgr, rect)
            if out is not None:
                palm_cands.append((out, rect))

        # --- choose the source (track-priority: no per-frame flipping) ---
        chosen: tuple[list[tuple[float, float]], str, float] | None = None
        chosen_rect: list[float] | None = None
        source: str | None = None
        if track_alive and track_out is not None:
            chosen, chosen_rect, source = track_out, track_rect, "track"
            track_pointing = is_pointing(track_out[0])
            self._np_run = 0 if track_pointing else self._np_run + 1
            for out, rect in palm_cands:
                d = hypot(rect[0] - track_rect[0], rect[1] - track_rect[1])
                cand_pointing = is_pointing(out[0])
                if d > NEAR_FACTOR * track_rect[2]:
                    # far candidate: a different object takes over when it
                    # is clearly better OR when it points and the track
                    # does not (pointing hand beats a face/junk anchor)
                    better = out[2] > track_out[2] + SWITCH_MARGIN
                    takeover = cand_pointing and not track_pointing
                    if better or takeover:
                        chosen, chosen_rect, source = out, rect, "switch"
                        break
                elif cand_pointing and not track_pointing and self._np_run >= NP_RUN_MIN:
                    # NEAR takeover: the hand crossed in front of the face
                    # (frame center) while the track sat on the face — the
                    # pointing hand reclaims its track. Only after a
                    # sustained non-pointing stretch, else the in-use
                    # hand's pointing flicker would switch sources.
                    chosen, chosen_rect, source = out, rect, "switch"
                    break
        elif track_alive and self._prev_rect is not None:
            # tracked crop missed (blur / fast swipe): a palm candidate near
            # the last position bridges the gap without resetting filters.
            # While the cursor is engaged, only pointing candidates may
            # bridge — a face anchor at frame center must not steal it.
            cands = palm_cands
            if prefer_pointing:
                pointing_only = [c for c in palm_cands if is_pointing(c[0][0])]
                if pointing_only:
                    cands = pointing_only
            best = max(cands, key=lambda c: self._eff(c[0]), default=None)
            if best is not None:
                out, rect = best
                d = hypot(rect[0] - self._prev_rect[0], rect[1] - self._prev_rect[1])
                if d <= NEAR_FACTOR * self._prev_rect[2]:
                    chosen, chosen_rect, source = out, rect, "track"
                elif out[2] >= self._class_score_threshold + SWITCH_MARGIN:
                    chosen, chosen_rect, source = out, rect, "switch"
        else:
            best = max(palm_cands, key=lambda c: self._eff(c[0]), default=None)
            if best is not None:
                chosen, chosen_rect, source = best[0], best[1], "init"

        if chosen is not None:
            self._miss = 0
            self._empty_palms = 0
            if source in ("switch", "init"):
                self.reset_smoothing()  # a different hand: no history to keep
            self._last_source = source
            # One Euro on the winner only — one update per landmark per
            # frame. Light touch: the screen-level filter does the main
            # smoothing, double-heavy filtering reads as sluggish lag.
            points = self._smooth(chosen[0], now_s)
            self._prev_points_norm = points
            if chosen_rect is None:
                chosen_rect, _ = rect_from_landmarks(
                    points, image_w, image_h, self._prev_dir, None
                )
            self._prev_rect = chosen_rect
            self._update_velocity((chosen_rect[0], chosen_rect[1]), now_s)

            results = [(points, chosen[1], chosen[2])]

            # --- template capture: only a confident POINTING hand ---
            # --- template capture/re-validation ---
            # Capture only when the main loop's debounce CONFIRMED pointing
            # (engaged), the pose points right now, the score is solid and
            # the landmark cloud actually fills the crop (a collapsed blob
            # must never be frozen). A sync frame that fails any of these
            # kills the template — it never survives on stale references
            # longer than REFIT_EVERY frames.
            if self._template:
                if (
                    engaged
                    and is_pointing(points)
                    and chosen[2] >= TEMPLATE_START_SCORE
                    and self._spread_ok(points, chosen_rect, image_w, image_h)
                ):
                    if self._tpl is None:
                        self._tpl = HandTemplateTracker(min_response=TEMPLATE_MIN_RESPONSE)
                    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    if not self._tpl.capture(
                        frame_gray,
                        (chosen_rect[0], chosen_rect[1]),
                        chosen_rect[2],
                        points,
                        chosen[1],
                        chosen[2],
                    ):
                        self._tpl = None
                elif self._tpl is not None:
                    self._tpl = None  # sync could not re-validate: full pipeline

            # --- static-object rejection (see constants above) ---
            if self._static_check(
                points, (chosen_rect[0], chosen_rect[1]), now_s, image_w, image_h
            ):
                results = []
        else:
            self._miss += 1
            if self._miss > TRACK_MISS_LIMIT:
                self._drop_track()
            results = []

        # --- phantom guard: palm sees NOTHING anywhere while a track lives ---
        if not palm_cands and not palm_rects:
            if track_alive and results:
                self._empty_palms += 1
                if self._empty_palms >= TRACK_EMPTY_PALM_LIMIT:
                    self._drop_track()
                    results = []
            else:
                self._empty_palms = 0

        return results
