"""neuroint cursor v2 — поднятый указательный палец ведёт курсор. Всё.

Run:  make run        (debug window)
      make headless   (без окна)
      make smoke      (12с самопроверка: камера+GPU+детект, мышь не трогает)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2

from neuroint_cursor.camera import Camera
from neuroint_cursor.config import Settings
from neuroint_cursor.cursor import CursorActuator
from neuroint_cursor.hands import OnnxHands
from neuroint_cursor.overlay import draw
from neuroint_cursor.pose import is_pointing

logger = logging.getLogger("neuroint_cursor")

MODELS = Path(__file__).resolve().parents[2] / "models"
PALM_MODEL = MODELS / "palm_detection_full_inf_post_192x192.onnx"
LANDMARK_MODEL = MODELS / "hand_landmark_sparse_Nx3x224x224.onnx"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="index-finger pointing -> cursor")
    ap.add_argument("--no-window", action="store_true", help="no debug window")
    ap.add_argument("--dry", action="store_true", help="log cursor target instead of moving")
    ap.add_argument("--smoke", type=float, default=0.0, help="run N seconds, print stats, exit")
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = Settings.load()
    if args.no_window:
        cfg.debug_window = False

    cam = Camera(cfg.camera_index, cfg.width, cfg.height, cfg.fps, cfg.mirror)
    hands = OnnxHands(
        PALM_MODEL,
        LANDMARK_MODEL,
        score_threshold=cfg.palm_score_threshold,
        class_score_threshold=cfg.landmark_score_threshold,
        tracking=cfg.tracking,
        clahe=cfg.clahe,
    )
    act = CursorActuator(cfg.sensitivity, cfg.min_cutoff, cfg.beta)
    logger.info(
        "screen %dx%d  active region %.2f of frame  provider %s",
        act.screen_w,
        act.screen_h,
        act.mapper.region_fraction,
        "DML" if "DmlExecutionProvider" in hands._landmark.get_providers() else "CPU",  # noqa: SLF001
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

                results = hands.detect(frame, prefer_pointing=active)
                result = results[0] if results else None
                if not results:
                    hands.reset_smoothing()

                pointing = result is not None and is_pointing(result[0])
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
                    fx, fy = result[0][8]  # index fingertip (already smoothed)
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
                        act.mapper.region(),
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
