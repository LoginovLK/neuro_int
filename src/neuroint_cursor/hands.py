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
  face), a still non-pinch object is quarantined, and while the
  cursor is engaged a far non-pinch candidate cannot steal the hand.

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

from neuroint_cursor.pose import is_pinch, palm_sane
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

# Bonus for a pinch candidate when CHOOSING among several acquisition
# candidates: the bundle hand must win over a junk anchor at equal scores.
PINCH_BONUS = 0.15

# Static-object rejection: shelf junk / furniture edges get detected as a
# palm and then tracked forever. A real hand always micro-sways, and a
# hand HELD deliberately still is in pinch pose (in use). So: an
# object that stays put for STATIC_SECONDS while never pinching is not
# a hand — drop it and remember a quarantine ZONE at that spot.
# "Stays put" = the rect center hovers near the window median OR the
# boxes keep overlapping (IoU): on textured junk the palm box hops
# between bright spots, centers spread wider than any radius, but the
# boxes still cover the same object — measured live (the skeleton that
# would not go away). Zones are PERSISTENT: the old 8s expiry made the
# junk re-appear in a 2s-visible/8s-hidden cycle; a zone is cleared only
# by a PINCH candidate at that spot (intentional use) or bypassed by
# a big close hand.
STATIC_SECONDS = 2.0
STATIC_RADIUS = 0.16      # median-center radius, fraction of min(frame_w, frame_h)
STATIC_IOU = 0.4          # median box overlap that also counts as "same object"
QUARANTINE_RADIUS = 0.08  # zone rejection radius, fraction of min(frame_w, frame_h)
ZONE_LIMIT = 6            # FIFO cap on remembered zones

# Shape lock (Lev's architecture): the landmark SHAPE is frozen once and
# re-anchored every frame on the live wrist/rotation/scale — the bones
# do not jitter frame to frame, only the rigid transform follows the
# hand. The shape re-freezes only when the live landmark cloud diverges
# from the rigid model by this fraction of the hand size (the pose
# actually changed).
SHAPE_REFIT = 0.35

# Still-PINCH sticker: a hallucination that reads pinch forever is
# exempt from the never-pinch quarantine — and it can even ENGAGE the
# cursor (measured live: "POINTING - cursor follows finger" on a bed
# corner). The discriminator a single frame cannot fake: a hand IN USE
# moves the cursor, so its fingertip wanders; a sticker is frozen. After
# STILL_PINCH_SECONDS of continuous pinch with a frozen fingertip
# (or a frozen box) the spot is locked out: control drops, candidates
# there are rejected. The lock opens when the box moves away / vanishes
# for LOCK_UNLOCK_SECONDS, a big close hand arrives, or a pinch
# candidate with a DIFFERENT fingertip appears (a real hand coming to
# that corner; the sticker's tip is frozen within the spread bound, so
# it can never satisfy the tip-difference test itself).
STILL_PINCH_SECONDS = 5.0
STILL_TIP_SPREAD = 0.02   # frozen-tip radius, fraction of min(frame_w, frame_h)
LOCK_TIP_DIFF = 0.06      # tip distance that proves "a different hand"
LOCK_IOU = 0.3            # box overlap that counts as "still at the spot"
LOCK_UNLOCK_SECONDS = 1.0 # spot unattended this long -> lock opens

# A candidate far from the last hand position (this factor of the last
# rect size) may not steal the track while the cursor is engaged and it
# is not pinching.
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
        face_model: str | Path | None = None,
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

        # Face gate (YuNet, OpenCV Zoo): the palm model fires on faces at
        # any threshold that still finds hands — measured live: quarantine
        # events at (0.50, 0.60) = Lev's face, repeated every ~2s, resetting
        # the bones each time. A real face detector kills the candidate
        # BEFORE the landmark call (also saves its retry-storm cost).
        self._face_det = None
        if face_model is not None and Path(face_model).exists():
            try:
                self._face_det = cv2.FaceDetectorYN.create(
                    str(face_model), "", (640, 360), score_threshold=0.6
                )
                logger.info("face gate: YuNet loaded")
            except Exception as exc:  # noqa: BLE001 - gate is optional
                logger.warning("face gate unavailable: %s", exc)

        self._filters: list[PointFilter2D] | None = None
        self._prev_points_norm: list[tuple[float, float]] | None = None
        self._prev_rect: list[float] | None = None
        self._miss = 0
        self._rot_fix = 0.0
        self._hist: deque[tuple[float, float, float, float]] = deque()
        self._tip_hist: deque[tuple[float, float, float]] = deque()
        self._shape: tuple[list[tuple[float, float]], float] | None = None
        self._zones: list[tuple[float, float]] = []
        self._lock: tuple[float, float, float, float, float] | None = None
        self._lock_seen = 0.0

    def _drop_track(self) -> None:
        self._prev_points_norm = None
        self._prev_rect = None
        self._miss = 0
        self._rot_fix = 0.0
        self._hist.clear()
        self._tip_hist.clear()
        self._shape = None
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
        """Effective score: pinch candidates outscore equal raw anchors."""
        return out[2] + (PINCH_BONUS if is_pinch(out[0]) else 0.0)

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

    def _shape_apply(
        self, raw: list[tuple[float, float]], image_w: int, image_h: int
    ) -> list[tuple[float, float]]:
        """Freeze the landmark SHAPE, rigidly re-anchor it every frame.

        Lev's architecture: the bones are not re-fit per frame — the
        frozen shape is translated onto the live wrist and rotated/
        scaled by the live wrist->middle-MCP transform. Only 3 degrees
        of freedom follow the CNN, so per-landmark jitter is gone and
        the skeleton cannot slide. Re-freezes only when the live cloud
        diverges from the rigid model beyond SHAPE_REFIT (the pose
        actually changed).
        """
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
            return raw
        offsets, ang0 = self._shape
        d = normalize_radians(ang - ang0)
        ca, sa = cos(d), sin(d)
        wx, wy = px[0]
        emitted = [
            (wx + hand * (ox * ca - oy * sa), wy + hand * (ox * sa + oy * ca))
            for ox, oy in offsets
        ]
        div = median(
            hypot(emitted[i][0] - px[i][0], emitted[i][1] - px[i][1])
            for i in range(21)
        ) / hand
        if div > SHAPE_REFIT:
            # pose change: re-freeze from the live cloud
            self._shape = (
                [((p[0] - px[0][0]) / hand, (p[1] - px[0][1]) / hand) for p in px],
                ang,
            )
            return raw
        return [(ex / image_w, ey / image_h) for ex, ey in emitted]

    def _track_tip(
        self,
        pinch_now: bool,
        raw_points: list[tuple[float, float]],
        now_s: float,
        image_w: int,
        image_h: int,
    ) -> None:
        """Fingertip history for the still-pinch sticker check."""
        if not pinch_now:
            self._tip_hist.clear()  # a broken pinch streak resets it
            return
        horizon = now_s - STILL_PINCH_SECONDS
        while self._tip_hist and self._tip_hist[0][0] < horizon:
            self._tip_hist.popleft()
        tip = raw_points[8]
        self._tip_hist.append((now_s, tip[0] * image_w, tip[1] * image_h))

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

    def _remember_zone(self, zx: float, zy: float) -> None:
        zone = (zx, zy)
        if zone not in self._zones:
            self._zones.append(zone)
            del self._zones[:-ZONE_LIMIT]

    def _static_check(
        self,
        points: list[tuple[float, float]],
        rect: list[float],
        now_s: float,
        image_w: int,
        image_h: int,
        pinch_now: bool,
    ) -> bool:
        """True when the object was just quarantined / locked out."""
        self._hist.append((now_s, rect[0], rect[1], float(rect[2])))
        horizon = now_s - STILL_PINCH_SECONDS
        while self._hist and self._hist[0][0] < horizon:
            self._hist.popleft()
        min_dim = min(image_w, image_h)

        # tier 1: still and NEVER pinching for 2s -> persistent zone
        if not pinch_now:
            recent = [r for r in self._hist if r[0] >= now_s - STATIC_SECONDS]
            if (
                len(recent) >= 15
                and recent[-1][0] - recent[0][0] >= STATIC_SECONDS * 0.8
            ):
                cxs = [c for _, c, _, _ in recent]
                cys = [c for _, _, c, _ in recent]
                med_cx, med_cy = median(cxs), median(cys)
                med_dist = hypot(cxs[-1] - med_cx, cys[-1] - med_cy)
                med_iou = median([self._iou(recent[-1], r) for r in recent])
                # Still by center OR by overlap: textured junk hops its box
                # between bright spots (centers spread, IoU stays high).
                if med_dist <= STATIC_RADIUS * min_dim or med_iou >= STATIC_IOU:
                    self._remember_zone(med_cx / image_w, med_cy / image_h)
                    self._hist.clear()
                    self._tip_hist.clear()
                    self._drop_track()
                    logger.info(
                        "static object at (%.2f, %.2f) quarantined",
                        med_cx / image_w,
                        med_cy / image_h,
                    )
                    return True

        # tier 2: still-PINCH sticker (5s frozen fingertip or frozen box)
        if (
            pinch_now
            and len(self._tip_hist) >= 30
            and self._tip_hist[-1][0] - self._tip_hist[0][0]
            >= STILL_PINCH_SECONDS * 0.8
        ):
            txs = [x for _, x, _y in self._tip_hist]
            tys = [y for _, _x, y in self._tip_hist]
            tip_spread = hypot(max(txs) - min(txs), max(tys) - min(tys))
            med_iou = median([self._iou(self._hist[-1], r) for r in self._hist])
            if tip_spread <= STILL_TIP_SPREAD * min_dim or med_iou >= STATIC_IOU:
                tip = self._tip_hist[-1]
                self._lock = (rect[0], rect[1], float(rect[2]), tip[1], tip[2])
                self._lock_seen = now_s
                self._hist.clear()
                self._tip_hist.clear()
                self._drop_track()
                logger.warning(
                    "still-pinch sticker locked out at (%.2f, %.2f)",
                    rect[0] / image_w,
                    rect[1] / image_h,
                )
                return True
        return False

    def detect(
        self, frame_bgr: np.ndarray, prefer_pinch: bool = False
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

        prefer_pinch: the cursor is engaged — a candidate far from the
        last hand position and NOT pinching is rejected (a face anchor at
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
        faces = self._face_boxes(frame_bgr)
        best = self._best_palm_candidate(
            frame_bgr, image_w, image_h, palm_thr, prefer_pinch, faces
        )
        if best is not None:
            out, rect = best
            shaped = self._shape_apply(out[0], image_w, image_h)
            points = self._smooth(shaped, now_s)
            self._prev_points_norm = points
            self._prev_rect = rect
            self._miss = 0
            pinch_now = is_pinch(points)
            self._track_tip(pinch_now, points, now_s, image_w, image_h)
            if self._static_check(points, rect, now_s, image_w, image_h, pinch_now):
                return []
            return [(points, out[1], out[2])]

        # --- fallback: palm missed, MediaPipe rect from previous landmarks ---
        if self._tracking and self._prev_points_norm is not None:
            rect = rect_from_landmarks(self._prev_points_norm, image_w, image_h)
            if rect is not None:
                out = self._landmark_tracked(frame_bgr, rect)
                if out is not None and self._on_crop(out[0], rect, image_w, image_h):
                    shaped = self._shape_apply(out[0], image_w, image_h)
                    points = self._smooth(shaped, now_s)
                    self._prev_points_norm = points
                    self._prev_rect = rect
                    self._miss = 0
                    pinch_now = is_pinch(points)
                    self._track_tip(pinch_now, points, now_s, image_w, image_h)
                    if self._static_check(
                        points, rect, now_s, image_w, image_h, pinch_now
                    ):
                        return []
                    return [(points, out[1], out[2])]

        # --- nothing this frame ---
        self._miss += 1
        if self._miss > TRACK_MISS_LIMIT:
            self._drop_track()
        return []

    def _face_boxes(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        """YuNet face boxes scaled to full-res (x, y, w, h); runs on 640x360."""
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

    @staticmethod
    def _on_face(
        rect: list[float], faces: list[tuple[float, float, float, float]]
    ) -> bool:
        """Palm candidate sitting ON a face: center inside the face box
        (10% margin) and face-sized. A real hand raised at the forehead
        keeps its center OUTSIDE the face box (verified on Lev's gun-pose
        frame)."""
        for fx, fy, fw, fh in faces:
            if (
                fx - 0.1 * fw <= rect[0] <= fx + 1.1 * fw
                and fy - 0.1 * fh <= rect[1] <= fy + 1.1 * fh
                and rect[2] <= 1.3 * fw
            ):
                return True
        return False

    def _best_palm_candidate(
        self,
        frame_bgr: np.ndarray,
        image_w: int,
        image_h: int,
        palm_thr: float,
        prefer_pinch: bool,
        faces: list[tuple[float, float, float, float]] | None = None,
    ) -> tuple[tuple[list[tuple[float, float]], str, float], list[float]] | None:
        """Palm rects -> landmark inferences -> best gated candidate."""
        now_s = time.monotonic()
        rectsn = self._cluster_filter(self._palm_rects(frame_bgr, palm_thr))
        min_dim = min(image_w, image_h)
        prev = self._prev_rect
        lock = self._lock
        best: tuple[tuple[list[tuple[float, float]], str, float], list[float]] | None = None
        best_eff = -1.0
        for rect in rectsn:
            # Face gate before anything expensive: no landmark call, no
            # tracking, no quarantine cycle on a face.
            if faces and self._on_face(rect, faces):
                continue
            # Suppression lock (still-pinch sticker): a candidate
            # sitting on the locked spot is checked after the landmark
            # pass — only its fingertip can prove it is a different hand.
            suspect = False
            if lock is not None:
                lcx, lcy, lsize = lock[0], lock[1], lock[2]
                overlaps = self._iou(
                    (0.0, rect[0], rect[1], float(rect[2])), (0.0, lcx, lcy, lsize)
                ) >= LOCK_IOU or hypot(rect[0] - lcx, rect[1] - lcy) <= STATIC_RADIUS * min_dim
                if overlaps:
                    self._lock_seen = now_s
                    suspect = True
            out = self._landmark_smart(frame_bgr, rect)
            if out is None:
                continue
            pinch = is_pinch(out[0])
            if suspect and self._lock is not None:
                if rect[2] > 0.18 * image_w:
                    self._lock = None  # a real close hand took the spot
                else:
                    tip = out[0][8]
                    tip_d = hypot(
                        tip[0] * image_w - self._lock[3], tip[1] * image_h - self._lock[4]
                    )
                    if pinch and tip_d > LOCK_TIP_DIFF * min_dim:
                        self._lock = None  # a different hand took the spot
                    else:
                        continue  # the sticker itself: rejected
            # Persistent quarantine zones: a non-pinch candidate inside a
            # remembered junk zone is dropped (a big close hand bypasses);
            # a PINCH candidate in a zone clears it.
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
                if pinch:
                    self._zones.remove(zone)
                else:
                    continue
            if prefer_pinch and prev is not None and not pinch:
                # Engaged cursor: a far non-pinch candidate must not
                # steal the hand (face anchor at frame center).
                d = hypot(rect[0] - prev[0], rect[1] - prev[1])
                if d > RECOVERY_NEAR_FACTOR * prev[2]:
                    continue
            eff = self._eff(out)
            if eff > best_eff:
                best = (out, rect)
                best_eff = eff
        # The lock opens when nothing sits on the spot for a second.
        if (
            self._lock is not None
            and self._lock_seen
            and now_s - self._lock_seen > LOCK_UNLOCK_SECONDS
        ):
            self._lock = None
        return best
