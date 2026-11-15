"""ONNX hand detection: the MediaPipe two-stage pipeline on the GPU.

Ports the PINTO0309/hand-gesture-recognition-using-onnx ONNX models through
onnxruntime DirectML, wired following the REFERENCE implementation for
these exact models (researched from the sources, not invented):

- PRIMARY (every frame): palm detection on the full frame, then the
  landmark model on the best palm rect. PINTO's reference app.py calls
  palm_detection unconditionally — no landmark->rect feedback loop. The
  landmark-rect feedback collapses with this sparse model (measured: a
  squashed output at score 0.964 feeds a garbage rect, rect shrinks
  165->15px in 10 iterations), while the palm detector has no feedback
  and cannot drift. MediaPipe landmark-tracks only because their model
  is robust AND they must save detector cost on phone CPUs.
- FALLBACK (palm missed): the MediaPipe rect derived from the previous
  landmarks (hand_landmarks_to_rect_calculator.cc + RectTransformation
  scale 2.0 / shift_y -0.1 / square_long, ported verbatim in tracking.py)
  with a sticky rotation fix for the model's squash instability.
- Crowded palm-box clusters are dropped (the palm model fires all over a
  face), a still non-pointing object is quarantined, and while the
  cursor is engaged a far non-pointing candidate cannot steal the hand.

Crop->frame decode is the exact inverse of the rotated-crop transform
(rotation included; verified empirically with a synthetic marker) — a
translation-only decode is off by sin(angle)*distance on tilted hands.
CLAHE runs on the palm frame and every crop (dark-room contrast).
Tilted hands: on a sanity failure (_landmark_smart) the rect is re-inferred
with rotated copies and the first SANE result wins (no best-by-ratio
fallback: a garbage accept would slide the bones).

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
from statistics import median

import cv2
import numpy as np
import onnxruntime as ort

from neuroint_cursor.pose import is_pointing, palm_sane
from neuroint_cursor.smoothing import PointFilter2D
from neuroint_cursor.tracking import (
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

# Bonus for a pointing candidate when CHOOSING among several acquisition
# candidates (MediaPipe has no such preference; the cursor use case does —
# the pointing hand must win over a face anchor at equal scores).
POINTING_BONUS = 0.15

# Static-object rejection: shelf junk / furniture edges get detected as a
# palm and then tracked forever. A real hand always micro-sways, and a
# hand HELD deliberately still is in pointing pose (in use). So: an
# object that stays put for STATIC_SECONDS while never pointing is not
# a hand — drop it and remember a quarantine ZONE at that spot.
# "Stays put" = the rect center hovers near the window median OR the
# boxes keep overlapping (IoU): on textured junk the palm box hops
# between bright spots, centers spread wider than any radius, but the
# boxes still cover the same object — measured live (the skeleton that
# would not go away). Zones are PERSISTENT: the old 8s expiry made the
# junk re-appear in a 2s-visible/8s-hidden cycle; a zone is cleared only
# by a POINTING candidate at that spot (intentional use) or bypassed by
# a big close hand.
STATIC_SECONDS = 2.0
STATIC_RADIUS = 0.16      # median-center radius, fraction of min(frame_w, frame_h)
STATIC_IOU = 0.4          # median box overlap that also counts as "same object"
QUARANTINE_RADIUS = 0.08  # zone rejection radius, fraction of min(frame_w, frame_h)
ZONE_LIMIT = 6            # FIFO cap on remembered zones

# A candidate far from the last hand position (this factor of the last
# rect size) may not steal the track while the cursor is engaged and it
# is not pointing.
RECOVERY_NEAR_FACTOR = 2.0

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
    """Crop a rotated rectangle (mirror padding only where it leaves the frame).

    Pads a LOCAL window around the box instead of the whole frame: the old
    full-frame copyMakeBorder cost ~10ms per landmark call, and the retry
    sweep runs it up to 5 times per frame (the live 14.5fps bug). Same
    geometry: the local window IS the axis-aligned bbox region the old
    code cropped out of the padded frame.
    """
    # rect comes in as float32: everything used for slicing must be int.
    cx, cy, rect_w, rect_h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    angle = float(rect[4])
    rect_tuple = ((cx, cy), (rect_w, rect_h), angle)
    box = cv2.boxPoints(rect_tuple)
    x_min, y_min = int(np.min(box[:, 0])), int(np.min(box[:, 1]))
    x_max, y_max = int(np.max(box[:, 0])), int(np.max(box[:, 1]))
    img_h, img_w = image.shape[:2]
    ix0, iy0 = max(x_min, 0), max(y_min, 0)
    ix1, iy1 = min(x_max, img_w), min(y_max, img_h)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return np.zeros((0, 0, 3), np.uint8)
    local = image[iy0:iy1, ix0:ix1]
    pad_top, pad_left = iy0 - y_min, ix0 - x_min
    pad_bottom, pad_right = y_max - iy1, x_max - ix1
    if pad_top or pad_bottom or pad_left or pad_right:
        # Mirror-reflection padding instead of black: a hand at the frame
        # edge gets a seamless continuation, so the landmark model keeps
        # its score (same BORDER_REFLECT_101 as before, just local).
        local = cv2.copyMakeBorder(
            local, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )

    height, width = local.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, int(angle), 1)
    abs_cos, abs_sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    matrix[0, 2] += bound_w / 2 - center[0]
    matrix[1, 2] += bound_h / 2 - center[1]
    rotated = cv2.warpAffine(local, matrix, (bound_w, bound_h))

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
        class_score_threshold: float = 0.25,
        tracking: bool = True,
        clahe: bool = True,
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
        self._class_score_threshold = class_score_threshold
        self._tracking = tracking
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

        self._filters: list[PointFilter2D] | None = None
        self._prev_points_norm: list[tuple[float, float]] | None = None
        self._prev_rect: list[float] | None = None
        self._miss = 0
        self._rot_fix = 0.0
        self._hist: deque[tuple[float, float, float, float]] = deque()
        self._zones: list[tuple[float, float]] = []

    def _drop_track(self) -> None:
        self._prev_points_norm = None
        self._prev_rect = None
        self._miss = 0
        self._rot_fix = 0.0
        self._hist.clear()
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
        padded, _ = _keep_aspect_resize_and_pad(frame_bgr, palm_hw_w, palm_hw_h)
        # CLAHE AFTER the 192x192 resize: same contrast lift, ~0.2ms
        # instead of ~10ms on the full 1280x720 frame (the old order
        # enhanced every pixel of the frame only to shrink it 7x).
        enhanced = self._enhance(padded) if self._clahe is not None else padded
        blob = enhanced / 255.0
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
        """Palm-path landmark inference with rotation retries on a squash.

        The sparse 224 landmark model intermittently outputs a SQUASHED
        hand (palm_ratio far outside the band) at high confidence on
        tilted palms — measured 0.34 ratio at score 0.964. A rotated crop
        fixes it; accepting the best-by-ratio garbage instead would make
        the choice flip between frames and slide the bones, so: sane or
        nothing.
        """
        base = self._landmark_from_rect(frame_bgr, rect, min_score)
        if base is None and min_score is not None:
            base = self._landmark_from_rect(
                frame_bgr, rect, min_score=self._class_score_threshold
            )
        if base is not None and palm_sane(base[0]):
            return base
        eff = min_score if base is not None else self._class_score_threshold
        # Sticky first: the offset that produced the last sane result is
        # reused (the same tilt recurs every frame; re-searching it
        # costs 4 inferences and cannot change the answer).
        if self._rot_fix != 0.0:
            rect2 = [rect[0], rect[1], rect[2], rect[3], rect[4] + self._rot_fix]
            out = self._landmark_from_rect(frame_bgr, rect2, min_score=eff)
            if out is not None and palm_sane(out[0]):
                return out
        for off in ROTATION_RETRIES:
            if off == self._rot_fix:
                continue
            rect2 = [rect[0], rect[1], rect[2], rect[3], rect[4] + off]
            out = self._landmark_from_rect(frame_bgr, rect2, min_score=eff)
            if out is not None and palm_sane(out[0]):
                self._rot_fix = off
                return out
        return None

    def _landmark_tracked(
        self, frame_bgr: np.ndarray, rect: list[float]
    ) -> tuple[list[tuple[float, float]], str, float] | None:
        """Tracking-path inference with a STICKY rotation fix.

        Same squash problem as the palm path, but here temporal
        consistency matters most: the offset that produced the last sane
        result is reused frame to frame (tried first), so the bones do
        not wander between differently-rotated solutions. Sane or
        nothing — a garbage frame becomes a miss and the recovery path
        takes over.
        """
        base = self._landmark_from_rect(frame_bgr, rect, min_score=TRACK_SCORE_MIN)
        if base is not None and palm_sane(base[0]):
            return base
        if self._rot_fix != 0.0:
            rect2 = [rect[0], rect[1], rect[2], rect[3], rect[4] + self._rot_fix]
            out = self._landmark_from_rect(frame_bgr, rect2, min_score=TRACK_SCORE_MIN)
            if out is not None and palm_sane(out[0]):
                return out
        for off in ROTATION_RETRIES:
            if off == self._rot_fix:
                continue
            rect2 = [rect[0], rect[1], rect[2], rect[3], rect[4] + off]
            out = self._landmark_from_rect(frame_bgr, rect2, min_score=TRACK_SCORE_MIN)
            if out is not None and palm_sane(out[0]):
                self._rot_fix = off
                return out
        return None

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
        """Winner-level One Euro — one update per landmark per frame.

        MediaPipe does the same (LandmarksSmoothingCalculator / velocity
        filter, a One-Euro-family filter). Light touch: the screen-level
        filter does the main smoothing.
        """
        if self._filters is None:
            self._filters = [PointFilter2D(min_cutoff=1.0, beta=0.02) for _ in range(21)]
        return [
            (float(sx), float(sy))
            for sx, sy in (
                self._filters[i].filter(x, y, now_s)
                for i, (x, y) in enumerate(raw_points)
            )
        ]

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        """Axis-aligned IoU of two (t, cx, cy, size) square boxes."""
        ax0, ax1 = a[1] - a[3] / 2, a[1] + a[3] / 2
        ay0, ay1 = a[2] - a[3] / 2, a[2] + a[3] / 2
        bx0, bx1 = b[1] - b[3] / 2, b[1] + b[3] / 2
        by0, by1 = b[2] - b[3] / 2, b[2] + b[3] / 2
        iw = min(ax1, bx1) - max(ax0, bx0)
        ih = min(ay1, by1) - max(ay0, by0)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = a[3] * a[3] + b[3] * b[3] - inter
        return inter / union if union > 0 else 0.0

    def _static_check(
        self,
        points: list[tuple[float, float]],
        rect: list[float],
        now_s: float,
        image_w: int,
        image_h: int,
    ) -> bool:
        """True when the object was just quarantined (still, never pointing)."""
        self._hist.append((now_s, rect[0], rect[1], float(rect[2])))
        horizon = now_s - STATIC_SECONDS
        while self._hist and self._hist[0][0] < horizon:
            self._hist.popleft()
        if (
            len(self._hist) < 15
            or self._hist[-1][0] - self._hist[0][0] < STATIC_SECONDS * 0.8
            or is_pointing(points)
        ):
            return False
        cxs = [c for _, c, _, _ in self._hist]
        cys = [c for _, _, c, _ in self._hist]
        med_cx, med_cy = median(cxs), median(cys)
        min_dim = min(image_w, image_h)
        med_dist = hypot(cxs[-1] - med_cx, cys[-1] - med_cy)
        med_iou = median([self._iou(self._hist[-1], r) for r in self._hist])
        # Still by center OR by overlap: textured junk hops its box
        # between bright spots (centers spread, IoU stays high).
        if med_dist > STATIC_RADIUS * min_dim and med_iou < STATIC_IOU:
            return False
        zone = (med_cx / image_w, med_cy / image_h)
        if zone not in self._zones:
            self._zones.append(zone)
            del self._zones[:-ZONE_LIMIT]
        self._hist.clear()
        self._drop_track()
        logger.info("static object at (%.2f, %.2f) quarantined", zone[0], zone[1])
        return True

    def detect(
        self, frame_bgr: np.ndarray, prefer_pointing: bool = False
    ) -> list[tuple[list[tuple[float, float]], str, float]]:
        """Full frame BGR -> list of (21 normalized landmarks, handedness, score).

        Primary source: the palm detector EVERY frame — this is the flow of
        the reference implementation for these exact ONNX models (PINTO0309
        hand-gesture-recognition-using-onnx app.py calls palm_detection
        unconditionally; its "tracking" is only palm-center ID matching).
        There is NO landmark->rect feedback loop: the sparse landmark model
        intermittently squashes (palm_ratio 0.34 at score 0.964 measured),
        and a feedback loop built on that output collapses (rect 165->15px
        in 10 iterations, measured). MediaPipe can landmark-track because
        their landmark model is robust AND they must save the detector cost
        on phone CPUs; on the RTX the detector is ~5ms and we can afford it.

        Fallback for the frames where the palm detector misses (motion
        blur, frame edge): the MediaPipe rect derived from the previous
        landmarks (hand_landmarks_to_rect_calculator.cc verbatim, see
        tracking.py) with the sticky rotation fix.

        prefer_pointing: the cursor is engaged — a candidate far from the
        last hand position and NOT pointing is rejected (a face anchor at
        frame center must not steal the hand mid-use).
        """
        image_h, image_w = frame_bgr.shape[:2]
        now_s = time.monotonic()

        # --- primary: palm detector every frame ---
        # ONE threshold (0.4) for both states. The old 0.3 "far hand"
        # threshold is dead: on Lev's face frame the palm model fires two
        # boxes that live ONLY at 0.3 (zero at 0.4), while the real hand
        # at working distance scores >=0.5 — the 0.3 knob was anchoring
        # the face, not finding far hands.
        palm_thr = self._score_threshold
        best = self._best_palm_candidate(
            frame_bgr, image_w, image_h, palm_thr, prefer_pointing
        )
        if best is not None:
            out, rect = best
            points = self._smooth(out[0], now_s)
            self._prev_points_norm = points
            self._prev_rect = rect
            self._miss = 0
            if self._static_check(points, rect, now_s, image_w, image_h):
                return []
            return [(points, out[1], out[2])]

        # --- fallback: palm missed, MediaPipe rect from previous landmarks ---
        if self._tracking and self._prev_points_norm is not None:
            rect = rect_from_landmarks(self._prev_points_norm, image_w, image_h)
            if rect is not None:
                out = self._landmark_tracked(frame_bgr, rect)
                if out is not None and self._on_crop(out[0], rect, image_w, image_h):
                    points = self._smooth(out[0], now_s)
                    self._prev_points_norm = points
                    self._prev_rect = rect
                    self._miss = 0
                    if self._static_check(points, rect, now_s, image_w, image_h):
                        return []
                    return [(points, out[1], out[2])]

        # --- nothing this frame ---
        self._miss += 1
        if self._miss > TRACK_MISS_LIMIT:
            self._drop_track()
        return []

    def _best_palm_candidate(
        self,
        frame_bgr: np.ndarray,
        image_w: int,
        image_h: int,
        palm_thr: float,
        prefer_pointing: bool,
    ) -> tuple[tuple[list[tuple[float, float]], str, float], list[float]] | None:
        """Palm rects -> landmark inferences -> best gated candidate."""
        rectsn = self._cluster_filter(self._palm_rects(frame_bgr, palm_thr))
        min_dim = min(image_w, image_h)
        prev = self._prev_rect
        best: tuple[tuple[list[tuple[float, float]], str, float], list[float]] | None = None
        best_eff = -1.0
        for rect in rectsn:
            out = self._landmark_smart(frame_bgr, rect)
            if out is None or not self._spread_ok(out[0], rect, image_w, image_h):
                continue
            pointing = is_pointing(out[0])
            # Persistent quarantine zones: a non-pointing candidate inside
            # a remembered junk zone is dropped (a big close hand — a real
            # hand near the camera — bypasses); a POINTING candidate in a
            # zone clears it (intentional use beats the quarantine).
            zone = next(
                (
                    z
                    for z in self._zones
                    if hypot(rect[0] - z[0] * image_w, rect[1] - z[1] * image_h)
                    <= QUARANTINE_RADIUS * min_dim
                ),
                None,
            )
            if zone is not None and not rect[2] > 0.18 * image_w:
                if pointing:
                    self._zones.remove(zone)
                else:
                    continue
            if prefer_pointing and prev is not None and not pointing:
                # Engaged cursor: a far non-pointing candidate must not
                # steal the hand (face anchor at frame center).
                d = hypot(rect[0] - prev[0], rect[1] - prev[1])
                if d > RECOVERY_NEAR_FACTOR * prev[2]:
                    continue
            eff = self._eff(out)
            if eff > best_eff:
                best = (out, rect)
                best_eff = eff
        return best
