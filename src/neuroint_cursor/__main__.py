"""neuroint cursor v3 — industry-standard gesture mouse.

Vocabulary (Ultraleap / Vision Pro / camera-mouse convention):

  open hand          -> MOVE   (cursor follows the index knuckle)
  pinch thumb-index  -> LMB press, release on open (click / hold = drag)
  pinch thumb-middle -> RMB press, release on open
  index+middle (V)   -> SCROLL: vertical hand travel = wheel steps
  fist / bundle      -> REST: cursor frozen, buttons up

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
from pynput.mouse import Button

from neuroint_cursor.camera import Camera
from neuroint_cursor.config import Settings
from neuroint_cursor.cursor import CursorActuator
from neuroint_cursor.overlay import draw
from neuroint_cursor.pose import (
    INDEX_MCP,
    is_move_pose,
    is_rest_pose,
    is_scroll_pose,
    pinch_dist,
    pinch_l_valid,
    pinch_r_valid,
)

logger = logging.getLogger("neuroint_cursor")

# canned-class score for pose CONFIRMATION only (Closed_Fist / Victory /
# Open_Palm are tilt-stable hints; clicks never come from classes)
OFFICIAL_CONFIRM = 0.5
# consecutive rest frames before a held button counts as dropped (a pose
# transition flickers 1-2 frames; five frames means a real fist)
REST_DROP_FRAMES = 5
# consecutive frames to enter/leave the scroll pose
SCROLL_FRAMES = 3
# a held button survives hand-tracking gaps up to this long (MediaPipe
# drops 1-3 frames of tracking under blur; a real hand-out releases here)
BUTTON_GRACE_S = 0.30
PINCH_EMA = 0.5

MODELS = Path(__file__).resolve().parents[2] / "models"
PALM_MODEL = MODELS / "palm_detection_full_inf_post_192x192.onnx"
LANDMARK_MODEL = MODELS / "hand_landmark_sparse_Nx3x224x224.onnx"
FACE_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
MP_MODEL = MODELS / "gesture_recognizer.task"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="gesture mouse: open hand + pinch")
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
        # OFFICIAL MediaPipe gesture recognizer (Google's own pipeline;
        # the manually ported ONNX models produced cramped skeletons)
        from neuroint_cursor.mp_hands import MediaPipeHands

        hands = MediaPipeHands(MP_MODEL, face_model=FACE_MODEL)
        logger.info("backend: mediapipe gesture recognizer (official)")
    except ImportError:
        from neuroint_cursor.hands import OnnxHands

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
    act = CursorActuator(
        cfg.sensitivity, cfg.min_cutoff, cfg.beta, actuate=not args.dry
    )
    logger.info("screen %dx%d  full-frame mapping", act.screen_w, act.screen_h)

    win = "neuroint cursor"
    if cfg.debug_window and not args.smoke:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(win, max(0, act.screen_w - 990), 60)
    else:
        win = None

    show_window = win is not None
    deadline = time.monotonic() + args.smoke if args.smoke else None

    frames = detected = engaged_frames = moved = 0
    engaged = False
    move_run = 0          # consecutive MOVE-pose frames (engage debounce)
    idle_run = 0          # consecutive REST/no-hand frames (disengage)
    scroll_run = 0        # consecutive SCROLL-pose frames
    scroll_active = False
    rest_run = 0          # consecutive REST frames (button drop)
    lmb_down = rmb_down = False
    last_fresh_t = time.monotonic()
    d_li_ema: float | None = None
    d_lm_ema: float | None = None
    scroll_y_ema: float | None = None
    status = "open hand to engage  |  fist = rest"
    fps_t = time.monotonic()
    fps_frames = 0
    fps = 0.0
    last_seq = -1
    canvas = None
    exit_code = 0

    def _release_buttons() -> None:
        nonlocal lmb_down, rmb_down
        if lmb_down:
            act.release(Button.left)
            logger.info("gesture: LMB up")
        if rmb_down:
            act.release(Button.right)
            logger.info("gesture: RMB up")
        lmb_down = rmb_down = False

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

                results = hands.detect(frame, prefer_pinch=engaged)
                result = results[0] if results else None
                # coast keeps the skeleton visible but is NOT a hand:
                # decisions need fresh detections. The window covers
                # MediaPipe's own 1-3 frame re-detection gaps (~100ms).
                # The ONNX backend has no _last_fresh: a returned hand IS
                # a fresh detection there.
                lf = getattr(hands, "_last_fresh", None)
                fresh = (now - lf) <= 0.15 if lf is not None else result is not None
                if fresh:
                    # the button grace counts from the last REAL detection,
                    # not from a coasting (stale) skeleton
                    last_fresh_t = now

                # ---- pose (fresh frames only) ----
                pose = "none"
                raw = getattr(hands, "last_raw", None)
                if fresh and result is not None:
                    if raw is None:
                        raw = result[0]
                    gname, gscore = getattr(hands, "last_gesture", (None, 0.0))
                    official = gscore >= OFFICIAL_CONFIRM
                    if is_scroll_pose(raw) or (official and gname == "Victory"):
                        pose = "scroll"
                    elif is_move_pose(raw) or (
                        official and gname in ("Open_Palm", "Pointing_Up")
                    ):
                        pose = "move"
                    elif is_rest_pose(raw) or (official and gname == "Closed_Fist"):
                        pose = "rest"
                    elif engaged:
                        pose = "move"  # ambiguous cloud: keep driving
                    else:
                        pose = "none"

                move_run = move_run + 1 if pose == "move" else 0
                idle_run = idle_run + 1 if pose in ("rest", "none") else 0
                rest_run = rest_run + 1 if pose == "rest" else 0

                # ---- engage / disengage ----
                if (
                    not engaged
                    and result is not None
                    and move_run >= cfg.debounce_frames
                ):
                    engaged = True
                    act.reset(result[0][INDEX_MCP][0], result[0][INDEX_MCP][1])
                    logger.info("engaged")
                elif engaged and idle_run >= cfg.debounce_off_frames:
                    engaged = False
                    _release_buttons()
                    act.scroll_end()
                    scroll_active = False
                    logger.info("released")

                # ---- held buttons ----
                if fresh and result is not None:
                    d_li = pinch_dist(raw, "index")
                    d_lm = pinch_dist(raw, "middle")
                    d_li_ema = d_li if d_li_ema is None else (
                        PINCH_EMA * d_li + (1 - PINCH_EMA) * d_li_ema
                    )
                    d_lm_ema = d_lm if d_lm_ema is None else (
                        PINCH_EMA * d_lm + (1 - PINCH_EMA) * d_lm_ema
                    )
                    if (
                        engaged
                        and not scroll_active
                        and not lmb_down
                        and not rmb_down
                    ):
                        # press fires on the threshold crossing: zero added
                        # latency (the Jota CHI'13 direct-manipulation rule)
                        if pinch_l_valid(raw, cfg.pinch_close):
                            act.press(Button.left)
                            lmb_down = True
                            logger.info("gesture: LMB down (pinch index)")
                        elif pinch_r_valid(raw, cfg.pinch_close, cfg.pinch_open):
                            act.press(Button.right)
                            rmb_down = True
                            logger.info("gesture: RMB down (pinch middle)")
                    if lmb_down and (d_li_ema > cfg.pinch_open or rest_run >= REST_DROP_FRAMES):
                        act.release(Button.left)
                        lmb_down = False
                        logger.info("gesture: LMB up")
                    if rmb_down and (d_lm_ema > cfg.pinch_open or rest_run >= REST_DROP_FRAMES):
                        act.release(Button.right)
                        rmb_down = False
                        logger.info("gesture: RMB up")
                elif now - last_fresh_t > BUTTON_GRACE_S:
                    _release_buttons()  # hand really gone: never leave a stuck button

                # ---- scroll ----
                scroll_run = scroll_run + 1 if pose == "scroll" else 0
                if (
                    engaged
                    and not lmb_down
                    and not rmb_down
                    and scroll_run >= SCROLL_FRAMES
                ):
                    if not scroll_active:
                        scroll_active = True
                        scroll_y_ema = raw[INDEX_MCP][1]
                        act.scroll_anchor(scroll_y_ema)
                        logger.info("gesture: scroll mode")
                elif scroll_active and scroll_run == 0:
                    scroll_active = False
                    scroll_y_ema = None
                    act.scroll_end()
                    logger.info("gesture: scroll end")
                if scroll_active and fresh and result is not None:
                    y = raw[INDEX_MCP][1]
                    scroll_y_ema = (
                        0.7 * scroll_y_ema + 0.3 * y if scroll_y_ema is not None else y
                    )
                    act.scroll_feed(scroll_y_ema)

                # ---- cursor movement (drag = pinch held + move) ----
                if engaged and result is not None and not scroll_active:
                    fx, fy = result[0][INDEX_MCP][0], result[0][INDEX_MCP][1]
                    if args.dry:
                        moved += 1
                        if moved % 15 == 1:
                            sx, sy = act.move(fx, fy, now)
                            logger.info("dry: cursor would move to %dx%d", sx, sy)
                    else:
                        act.move(fx, fy, now)
                        moved += 1

                if lmb_down:
                    status = "LMB (pinch index) - hold & move = drag"
                elif rmb_down:
                    status = "RMB (pinch middle)"
                elif scroll_active:
                    status = "SCROLL - two fingers, move hand up/down"
                elif engaged:
                    status = "MOVE - open hand  |  pinch = click  |  fist = rest"
                else:
                    status = "open hand to engage  |  fist = rest"

                frames += 1
                detected += 1 if result is not None else 0
                engaged_frames += 1 if engaged else 0
                fps_frames += 1
                if now - fps_t >= 1.0:
                    fps = fps_frames / (now - fps_t)
                    fps_frames = 0
                    fps_t = now
                    logger.info(
                        "fps %.1f  hands %s  engaged %s  lmb %s  rmb %s  scroll %s  sens %.1f",
                        fps,
                        bool(result),
                        engaged,
                        lmb_down,
                        rmb_down,
                        scroll_active,
                        act.sensitivity,
                    )

                if show_window:
                    canvas = draw(
                        frame,
                        result,
                        fps,
                        act.sensitivity,
                        status,
                        active=engaged,
                        lmb=lmb_down,
                        rmb=rmb_down,
                        scroll=scroll_active,
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
        _release_buttons()
        act.release_all()
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
                    "engaged_frames": engaged_frames,
                    "would_move_or_moved": moved,
                    "fps_avg": round(fps, 1),
                }
            )
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
