"""ONNX hand detection: the MediaPipe two-stage pipeline on the GPU.

Ports the PINTO0309/hand-gesture-recognition-using-onnx pipeline:
  1. palm_detection_full_inf_post_192x192.onnx — palm detector with NMS
     built into the ONNX graph
  2. hand_landmark_sparse_Nx3x224x224.onnx — 21 landmarks per hand

Runs through onnxruntime DirectML on the RTX GPU (~4-5ms/frame). Every
decode fix from the old pipeline is preserved here verbatim:
- rr_cx is already normalized to the frame (double-normalization bug)
- rr_cy un-pad needs the /scale step (hand drifting 200px bug)
- BORDER_REFLECT_101 padding in rotate_and_crop (edge-of-frame bug)
- int() casts on the float32 rect before numpy slicing
- run(...)[0] unpack (not [0][0]) for the [N,8] palm output
"""

from __future__ import annotations

import logging
import time
from math import atan2, cos, floor, pi, sin
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from neuroint_cursor.smoothing import PointFilter2D

logger = logging.getLogger(__name__)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _normalize_radians(angle: float) -> float:
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))


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
        score_threshold: float = 0.3,
        class_score_threshold: float = 0.25,
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
        self._smoothing: list[PointFilter2D] | None = None
        self._smooth_time: float | None = None

    def detect(
        self, frame_bgr: np.ndarray
    ) -> list[tuple[list[tuple[float, float]], str, float]]:
        """Full frame BGR -> list of (21 normalized landmarks, handedness, score)."""
        image_h, image_w = frame_bgr.shape[:2]

        # --- stage 1: palm detection ---
        palm_hw_w, palm_hw_h = self._palm_hw[1], self._palm_hw[0]
        padded, _ = _keep_aspect_resize_and_pad(frame_bgr, palm_hw_w, palm_hw_h)
        blob = padded / 255.0
        blob = blob[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        boxes = self._palm.run(self._palm_out, {self._palm_input: blob})[0]

        rects: list[list[float]] = []
        for box in boxes:
            score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = box
            if score <= self._score_threshold or box_size <= 0:
                continue
            kp02_y = kp2_y - kp0_y
            kp02_x = kp2_x - kp0_x
            rr_size = 2.9 * box_size
            rotation = _normalize_radians(0.5 * pi - atan2(-kp02_y, kp02_x))
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

        if not rects:
            return []

        # --- stage 2: landmarks per palm crop ---
        results: list[tuple[list[tuple[float, float]], str, float]] = []
        for rect in rects:
            crop = _rotate_and_crop(frame_bgr, np.asarray(rect, dtype=np.float32))
            if crop.size == 0:
                continue
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
            if float(hand_scores[0][0]) < self._class_score_threshold:
                continue

            lms = xyz21[0].reshape(21, 3) / self._lm_hw[0]
            xy = np.asarray([[p[0], p[1]] for p in lms], dtype=np.float32)
            xy[:, 0] = (xy[:, 0] * self._lm_hw[1] - lm_pad_w_half) / max(scale_w, 1e-6)
            xy[:, 1] = (xy[:, 1] * self._lm_hw[0] - lm_pad_h_half) / max(scale_h, 1e-6)
            # Crop-local pixels -> full frame pixels (rect offset) -> normalized.
            palm_score = float(score)
            cx, cy, rect_w, rect_h = rect[0], rect[1], rect[2], rect[3]
            xy[:, 0] = (xy[:, 0] + cx - rect_w // 2) / image_w
            xy[:, 1] = (xy[:, 1] + cy - rect_h // 2) / image_h

            # Display-only label: frames are mirrored at capture, so the
            # model's handedness is not used for control logic anywhere.
            handedness = "Right" if float(handed[0][0]) > 0.5 else "Left"

            # One Euro smoothing per landmark: kills jitter without adding
            # noticeable latency to fast pointing.
            now_s = time.monotonic()
            if self._smoothing is None or self._smooth_time is None:
                self._smoothing = [PointFilter2D(min_cutoff=0.7, beta=0.015) for _ in range(21)]
                self._smooth_time = now_s
            self._smooth_time = now_s
            points = [
                (float(sx), float(sy))
                for (sx, sy) in (
                    self._smoothing[i].filter(float(x), float(y), now_s)
                    for i, (x, y) in enumerate(xy)
                )
            ]
            results.append((palm_score, points, handedness))

        # Best hand first.
        results.sort(key=lambda r: r[0], reverse=True)
        return [(points, handedness, palm_score) for palm_score, points, handedness in results]

    def reset_smoothing(self) -> None:
        """Drop filter history (call when the hand leaves the frame)."""
        self._smoothing = None
        self._smooth_time = None
