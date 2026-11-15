"""neuroint cursor v2 — поднятый указательный палец ведёт курсор. Всё.

Run:  make run        (debug window)
      make headless   (без окна)
      make smoke      (12с самопроверка: камера+GPU+детект, мышь не трогает)
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )
    args = parse_args()
    cfg = Settings.load()
    if args.no_window:
        cfg.debug_window = False

    cam = Camera(cfg.camera_index, cfg.width, cfg.height, cfg.fps, cfg.mirror)
    if DWPOSE_DET.exists() and DWPOSE_POSE.exists():
        # researched replacement (Lev): DWPose RTMPose wholebody — hands
        # an order better than the MediaPipe-class sparse models, real-
        # time on the 3090.
        from neuroint_cursor.dwpose_hands import DWPoseHands

        hands = DWPoseHands(DWPOSE_DET, DWPOSE_POSE)  # type: ignore[assignment]
        logger.info("backend: DWPose wholebody (yolox + rtmpose-133)")
    else:
        hands = OnnxHands(
            PALM_MODEL,
            LANDMARK_MODEL,
            score_threshold=cfg.palm_score_threshold,
            class_score_threshold=cfg.landmark_score_threshold,
            tracking=cfg.tracking,
            clahe=cfg.clahe,
            face_model=FACE_MODEL,
        )
        logger.info("backend: palm+landmark (dwpose models not found)")
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
                if not results:
                    pass  # filters reset inside the backend on track loss

                pointing = result is not None and is_pinch(result[0])
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
