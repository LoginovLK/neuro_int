"""DWPose hand backend: YOLOX person detector + RTMPose wholebody 133.

Researched replacement for the MediaPipe-style sparse pipeline (Lev: the
bones quality is bad, use the GPU). Sources:
- DWPose (ControlNet annotator, distilled RTMPose wholebody) — 133
  keypoints including 21 per hand, ONNX, real-time on GPU; hands are
  the weak spot of MediaPipe-class models and DWPose beats it.
- rtmlib (Tau-J) — the lightweight ONNX runtime reference for these
  models; decode logic here follows its top-down pipeline: detector
  (YOLOX, person class) -> affine crop -> SimCC argmax -> inverse affine.

Same detect() interface as OnnxHands: [(21 normalized points, handedness,
score)], so pose/mapping/overlay keep working. The pinch gate, One Euro
smoothing, shape lock (frozen skeleton re-anchored rigidly) and the
still-pinch sticker lockout are carried over.

COCO-WholeBody hand index layout is MediaPipe-compatible (0 wrist,
1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky), so
HAND_CONNECTIONS stay valid.
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
import onnxruntime as ort

from neuroint_cursor.pose import is_pinch
from neuroint_cursor.smoothing import PointFilter2D

logger = logging.getLogger(__name__)

# SimCC decode: heatmaps are split at 2x the input resolution.
SIMCC_SPLIT = 2.0
PERSON_THR = 0.35
KPT_THR = 0.30
# COCO-WholeBody: left hand keypoints 91..111, right hand 112..132.
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)

STILL_PINCH_SECONDS = 5.0
STILL_TIP_SPREAD = 0.02   # frozen bundle radius, fraction of min(frame_w, frame_h)
LOCK_UNLOCK_PX = 0.06     # bundle moved this far -> unlock, fraction of min_dim

SHAPE_REFIT = 0.35


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.45) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


class DWPoseHands:
    """Person detector + wholebody pose -> one hand's 21 keypoints."""

    def __init__(
        self,
        det_model: str | Path,
        pose_model: str | Path,
        clahe: bool = False,
    ) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self._det = ort.InferenceSession(str(det_model), options, providers=providers)
        self._pose = ort.InferenceSession(str(pose_model), options, providers=providers)
        logger.info("dwpose providers: %s", self._pose.get_providers())

        self._det_in = self._det.get_inputs()[0].name
        self._det_out = [o.name for o in self._det.get_outputs()]
        self._pose_in = self._pose.get_inputs()[0].name
        self._pose_out = [o.name for o in self._pose.get_outputs()]
        self._det_hw = (640, 640)                      # YOLOX input
        self._pose_hw = (384, 288)                     # RTMPose (H, W)

        self._filters: list[PointFilter2D] | None = None
        self._shape: tuple[list[tuple[float, float]], float] | None = None
        self._tip_hist: deque[tuple[float, float, float]] = deque()
        self._lock: tuple[float, float, float, float] | None = None
        self._lock_seen = 0.0

    # ---------------- detector ----------------

    def _detect_persons(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        """YOLOX: letterbox 640x640, decode anchors, person-class NMS."""
        image_h, image_w = frame_bgr.shape[:2]
        dh, dw = self._det_hw
        scale = min(dw / image_w, dh / image_h)
        nw, nh = int(image_w * scale), int(image_h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        padded = np.full((dh, dw, 3), 114, np.uint8)
        padded[:nh, :nw] = resized
        blob = padded[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        raw = self._det.run(self._det_out, {self._det_in: blob})[0][0]  # [8400, 85]

        # YOLOX anchor decode (official demo): grids + strides per level.
        grids: list[np.ndarray] = []
        strides: list[np.ndarray] = []
        for stride in (8, 16, 32):
            hs, ws = dh // stride, dw // stride
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grid = np.stack((xv, yv), 2).reshape(-1, 2).astype(np.float32)
            grids.append(grid)
            strides.append(np.full((grid.shape[0], 2), stride, np.float32))
        grid = np.concatenate(grids)
        stride_m = np.concatenate(strides)
        raw[:, :2] = (raw[:, :2] + grid) * stride_m
        raw[:, 2:4] = np.exp(raw[:, 2:4]) * stride_m

        boxes_xywh = raw[:, :4]
        boxes = np.empty_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        person_scores = raw[:, 4] * raw[:, 5]  # objectness * person class
        mask = person_scores > PERSON_THR
        if not mask.any():
            return []
        idxs = _nms(boxes[mask], person_scores[mask])
        picked = np.nonzero(mask)[0][idxs]
        out = []
        for i in picked:
            x1, y1, x2, y2 = boxes[i] / scale  # undo letterbox
            out.append((float(x1), float(y1), float(x2), float(y2)))
        return out

    # ---------------- pose ----------------

    def _affine_crop(self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]):
        """RTMPose top-down affine: bbox*1.25 fitted to the input aspect."""
        ih, iw = frame_bgr.shape[:2]
        ph, pw = self._pose_hw
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(iw), x2), min(float(ih), y2)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pw_full, ph_full = (x2 - x1) * 1.25, (y2 - y1) * 1.25
        if pw_full / ph_full > pw / ph:
            ph_full = pw_full * ph / pw
        else:
            pw_full = ph_full * pw / ph
        src = np.float32(
            [[cx, cy], [cx + pw_full / 2, cy], [cx, cy + ph_full / 2]]
        )
        dst = np.float32([[pw / 2, ph / 2], [pw, ph / 2], [pw / 2, ph]])
        affine = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(frame_bgr, affine, (pw, ph))
        return crop, affine

    def _run_pose(
        self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """133 keypoints in frame pixels + per-keypoint scores [0..1].

        SimCC decode (rtmlib reference): the X coordinate comes from the
        x-grid argmax and the Y coordinate INDEPENDENTLY from the y-grid
        argmax — they are separate 1D distributions per keypoint.
        """
        crop, affine = self._affine_crop(frame_bgr, bbox)
        blob = crop[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        simcc_x, simcc_y = self._pose.run(self._pose_out, {self._pose_in: blob})
        val_x, x_loc = np.max(simcc_x[0], axis=1), np.argmax(simcc_x[0], axis=1)
        val_y, y_loc = np.max(simcc_y[0], axis=1), np.argmax(simcc_y[0], axis=1)
        pts = np.stack(
            [x_loc.astype(np.float32), y_loc.astype(np.float32)], axis=1
        ) / SIMCC_SPLIT  # crop pixels
        ones = np.ones((133, 1), np.float32)
        inv = cv2.invertAffineTransform(affine)
        pts_img = (np.hstack([pts, ones]) @ inv.T).astype(np.float32)
        scores = (val_x + val_y) / 2
        return pts_img, scores

    # ---------------- frozen shape (Lev's architecture) ----------------

    def _shape_apply(
        self, raw: list[tuple[float, float]], image_w: int, image_h: int
    ) -> list[tuple[float, float]]:
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
        d = ang - ang0
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
            self._shape = (
                [((p[0] - px[0][0]) / hand, (p[1] - px[0][1]) / hand) for p in px],
                ang,
            )
            return raw
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
        """True when the still-pinch sticker lockout suppresses this hand."""
        min_dim = min(image_w, image_h)
        cx, cy = pinch_center_px(points, image_w, image_h)
        if self._lock is not None:
            moved = hypot(cx - self._lock[0], cy - self._lock[1])
            if not pinch_now or moved > LOCK_UNLOCK_PX * min_dim:
                self._lock = None  # the hand started moving / pose changed
            elif now_s - self._lock_seen > 0.5:
                self._lock_seen = now_s
                return True  # still frozen: keep control off
            else:
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
                spread = hypot(max(xs) - min(xs), max(ys) - min(ys))
                if spread <= STILL_TIP_SPREAD * min_dim:
                    self._lock = (cx, cy, 0.0, 0.0)
                    self._lock_seen = now_s
                    self._tip_hist.clear()
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

        persons = self._detect_persons(frame_bgr)
        if not persons:
            self._shape = None
            self.reset_filters()
            return []
        # the biggest person on screen
        x1, y1, x2, y2 = max(
            persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
        )
        keypoints, scores = self._run_pose(frame_bgr, (x1, y1, x2, y2))

        candidates = []
        for sl, handed in ((LEFT_HAND, "Left"), (RIGHT_HAND, "Right")):
            kp = keypoints[sl]
            sc = scores[sl]
            visible = sc > KPT_THR
            if int(visible.sum()) < 12:
                continue
            pts21 = [
                (float(kp[i][0]) / image_w, float(kp[i][1]) / image_h)
                for i in range(21)
            ]
            candidates.append(
                (float(visible.mean()), handed, pts21, float(visible.mean()))
            )
        if not candidates:
            self._shape = None
            self.reset_filters()
            return []
        if prefer_pinch:
            pinched = [c for c in candidates if is_pinch(c[2])]
            if pinched:
                candidates = pinched
        _, handed, pts21, score = max(candidates, key=lambda c: c[0])

        shaped = self._shape_apply(pts21, image_w, image_h)
        points = self._smooth(shaped, now_s)
        pinch_now = is_pinch(points)
        if self._sticker_check(points, pinch_now, now_s, image_w, image_h):
            return []
        return [(points, handed, score)]

    def reset_filters(self) -> None:
        self._filters = None


def pinch_center_px(
    pts_norm: list[tuple[float, float]], image_w: int, image_h: int
) -> tuple[float, float]:
    from neuroint_cursor.pose import pinch_center

    fx, fy = pinch_center(pts_norm)
    return fx * image_w, fy * image_h
