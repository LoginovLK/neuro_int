"""neuroint cursor v2 — поднятый указательный палец ведёт курсор. Всё.

Run:  make run        (debug window)
      make headless   (без окна)
      make smoke      (12с самопроверка: камера+GPU+детект, мышь не трогает)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import logging.handlers
import queue as queue_mod
import threading
import time
from pathlib import Path

import cv2

from neuroint_cursor.camera import Camera
from neuroint_cursor.config import Settings
from neuroint_cursor.cursor import CursorActuator
from neuroint_cursor.hands import OnnxHands
from neuroint_cursor.overlay import draw
from neuroint_cursor.pose import is_pinch, pinch_center

logger = logging.getLogger("neuroint_cursor")

MODELS = Path(__file__).resolve().parents[2] / "models"
PALM_MODEL = MODELS / "palm_detection_full_inf_post_192x192.onnx"
LANDMARK_MODEL = MODELS / "hand_landmark_sparse_Nx3x224x224.onnx"
FACE_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
MP_MODEL = MODELS / "hand_landmarker.task"
DWPOSE_DET = MODELS / "yolox_l.onnx"
DWPOSE_POSE = MODELS / "dw-ll_ucoco_384.onnx"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="index-finger pointing -> cursor")
    ap.add_argument("--no-window", action="store_true", help="no debug window")
    ap.add_argument("--dry", action="store_true", help="log cursor target instead of moving")
    ap.add_argument("--smoke", type=float, default=0.0, help="run N seconds, print stats, exit")
    return ap.parse_args()


def main() -> int:
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "neuroint.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    # non-blocking: records go through a real queue.Queue to a background
    # thread - a sync disk flush inside the 30fps loop caused freezes
    log_queue = queue_mod.Queue()
    queue_handler = logging.handlers.QueueHandler(log_queue)
    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=False
    )
    listener.start()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[queue_handler],
    )
    args = parse_args()
    cfg = Settings.load()
    if args.no_window:
        cfg.debug_window = False

    # every mp.Image allocation feeds the Python GC; after ~a minute of
    # 30fps allocations a generation collect in the hot loop freezes the
    # cursor - keep gen-0 sweeps on a background timer instead
    def _gc_sweep() -> None:
        while True:
            time.sleep(2.5)
            gc.collect(0)

    threading.Thread(target=_gc_sweep, name="gc-sweep", daemon=True).start()

    cam = Camera(cfg.camera_index, cfg.width, cfg.height, cfg.fps, cfg.mirror)
    try:
        # OFFICIAL MediaPipe hand landmarker (Google's own palm-detect ->
        # landmark-track pipeline, preprocessing matched to training).
        # The manually ported ONNX models produced cramped, twitchy
        # skeletons; this is the reference quality Lev's original v1 had.
        from neuroint_cursor.mp_hands import MediaPipeHands

        hands = MediaPipeHands(MP_MODEL, face_model=FACE_MODEL)  # type: ignore[assignment]
        logger.info("backend: mediapipe hand landmarker (official)")
    except ImportError:
        hands = OnnxHands(
            PALM_MODEL,
            LANDMARK_MODEL,
            score_threshold=cfg.palm_score_threshold,
            class_score_threshold=cfg.landmark_score_threshold,
            tracking=cfg.tracking,
            clahe=cfg.clahe,
            face_model=FACE_MODEL,
        )
        logger.info("backend: palm-first onnx (mediapipe package not installed)")
    act = CursorActuator(cfg.sensitivity, cfg.min_cutoff, cfg.beta)
    logger.info(
        "screen %dx%d  full-frame mapping",
        act.screen_w,
        act.screen_h,
    )

    win = "neuroint cursor"
    if cfg.debug_window and not args.smoke:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(win, max(0, act.screen_w - 990), 60)
    else:
        win = None

    show_window = win is not None
    deadline = time.monotonic() + args.smoke if args.smoke else None

    frames = detected = pointing_frames = moved = 0
    active = False
    on_counter = off_counter = 0
    fps_t = time.monotonic()
    fps_frames = 0
    fps = 0.0
    last_seq = -1
    canvas = None
    exit_code = 0

    try:
        while True:
            frame, seq = cam.read()
            if frame is None:
                time.sleep(0.02)
                if deadline is not None and time.monotonic() > deadline + 5:
                    logger.error("smoke: no frames from camera")
                    exit_code = 1
                    break
                continue

            if seq != last_seq:
                last_seq = seq
                now = time.monotonic()

                results = hands.detect(frame, prefer_pinch=active)
                result = results[0] if results else None
                # coast keeps the skeleton visible but is NOT a hand:
                # control treats stale results as "no hand". The window
                # must cover MediaPipe's own 1-3 frame re-detection gaps
                # (measured ~100ms) or the cursor freezes every few
                # seconds; a real hand-out still releases via the
                # off-debounce right after.
                fresh = (now - getattr(hands, "_last_fresh", now)) <= 0.15

                pointing = fresh and result is not None and is_pinch(result[0])
                # Debounce both edges: ON after a few straight-pointing frames,
                # OFF only after a grace period (single dropped detections must
                # not stutter the cursor).
                on_counter = on_counter + 1 if pointing else 0
                off_counter = off_counter + 1 if not pointing else 0
                if not active and on_counter >= cfg.debounce_frames:
                    active = True
                    act.reset()
                elif active and off_counter >= cfg.debounce_off_frames:
                    active = False

                if active and result is not None:
                    fx, fy = pinch_center(result[0])  # the bundle center
                    sx, sy = act.mapper.map(fx, fy)
                    if args.dry:
                        moved += 1
                        if moved % 15 == 1:
                            logger.info("dry: cursor would move to %dx%d", sx, sy)
                    else:
                        act.move(fx, fy, now)
                        moved += 1

                frames += 1
                detected += 1 if result is not None else 0
                pointing_frames += 1 if pointing else 0
                fps_frames += 1
                if now - fps_t >= 1.0:
                    fps = fps_frames / (now - fps_t)
                    fps_frames = 0
                    fps_t = now
                    logger.info(
                        "fps %.1f  hands %s  pointing %s  sens %.1f",
                        fps,
                        bool(result),
                        active,
                        act.sensitivity,
                    )

                if show_window:
                    canvas = draw(
                        frame,
                        result,
                        fps,
                        act.sensitivity,
                        active,
                    )

            if show_window and canvas is not None:
                cv2.imshow(win, cv2.resize(canvas, (960, 540)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("+"), ord("=")):
                    act.set_sensitivity(act.sensitivity + 0.2)
                    cfg.sensitivity = act.sensitivity
                    cfg.save()
                    logger.info("sensitivity %.1f (saved)", act.sensitivity)
                if key in (ord("-"), ord("_")):
                    act.set_sensitivity(act.sensitivity - 0.2)
                    cfg.sensitivity = act.sensitivity
                    cfg.save()
                    logger.info("sensitivity %.1f (saved)", act.sensitivity)

            if deadline is not None and time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        listener.stop()
        if win is not None:
            cv2.destroyAllWindows()

    if args.smoke:
        print(
            json.dumps(
                {
                    "frames": frames,
                    "detected": detected,
                    "pointing_frames": pointing_frames,
                    "would_move_or_moved": moved,
                    "fps_avg": round(fps, 1),
                }
            )
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
