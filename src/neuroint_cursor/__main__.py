"""neuroint cursor v3 — industry-standard gesture mouse.

Vocabulary (Vision Pro convention — the hand is always active, there is
NO rest pose; shape flicker can never freeze the cursor):

  any hand in frame  -> MOVE   (cursor follows the index knuckle)
  pinch thumb-index  -> LMB press, release on open (click / hold = drag)
  index+middle (V)   -> SCROLL: vertical hand travel = wheel steps
  fist / bundle      -> cannot START a click (the d46 guard gates the
                        press); a click already held lives until the
                        pinch opens or the hand leaves the frame
  hand out of frame  -> release everything, cursor frozen

  RMB pinch gesture REMOVED (live 2026-11-22: this hand's bundle runs
  d412 < d48, so every dominance gate fired false RMBs inside bundles —
  15/15 events were accidents). Vision Pro ships without a pinch-RMB
  for the same reason: one pinch primitive only.

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
from collections import deque
from pathlib import Path

import cv2
from pynput.mouse import Button

from neuroint_cursor.camera import Camera
from neuroint_cursor.config import Settings
from neuroint_cursor.cursor import CursorActuator
from neuroint_cursor.overlay import draw
from neuroint_cursor.pose import (
    INDEX_MCP,
    is_scroll_pose,
    palm_center,
    palm_length,
    pinch_dist,
    pinch_l_fire,
    thumb_pip_dist,
)

logger = logging.getLogger("neuroint_cursor")

# canned-class score for pose CONFIRMATION only (Victory / Open_Palm are
# tilt-stable hints; clicks never come from classes)
OFFICIAL_CONFIRM = 0.5
# A held button is dropped ONLY by the pinch opening or the hand leaving
# the frame — NOT by the hand closing tighter. Measured live: a natural
# strong pinch evolves into a gathered bundle (d46 collapses to ~0.1)
# and the old fist-signature drop killed every hold 140-300ms in; the
# entry d46 gate alone keeps a fist from ever STARTING a click.
# Scroll pose: consecutive frames to enter (transition shapes flash two
# extended fingers; a real scroll intent holds)...
SCROLL_FRAMES = 5
# ...and to LEAVE again — single misreads must not chatter scroll on/off
# (measured: a 94ms on/off flicker at the old value of 3)
SCROLL_OFF_FRAMES = 6
# a held button survives hand-tracking gaps up to this long (MediaPipe
# drops 1-3 frames of tracking under blur; a real hand-out releases here)
BUTTON_GRACE_S = 0.30
PINCH_EMA = 0.5
# after a press, ignore new presses for this long (anti-bounce: a human
# double-click re-pinches at ~0.25-0.4s, jitter re-fires at 30-60ms)
PRESS_REFRACTORY_S = 0.15
# a press shorter than this stretches to it, so the OS always sees a
# real click instead of a sub-frame blip
MIN_HOLD_S = 0.08
# no press within this window after engage: the hand often enters the
# frame already half-closed (measured: a false LMB 22ms after engage)
ENGAGE_GRACE_S = 0.30
# palm-length EMA for distance normalization: the per-frame projected
# palm collapses 2-4x when the hand turns edge-on (measured d46 swinging
# 0.24 -> 2.85 within a second), dragging every threshold along; a slow
# denominator keeps the units stable for the lifetime of a gesture
PALM_EMA = 0.1


def _med3(buf: deque) -> float:
    """Median of the last three samples — one-frame landmark spikes
    (d48 jumping 0.24 -> 0.80 -> 0.24, measured live) collapse before
    any threshold sees them, at +2 frames of latency (66ms, imperceptible)."""
    return sorted(buf)[len(buf) // 2]

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
    move_run = 0          # consecutive hand-present frames (engage debounce)
    idle_run = 0          # consecutive no-hand frames (disengage)
    scroll_run = 0        # consecutive SCROLL-pose frames
    scroll_off_run = 0    # consecutive non-scroll frames (scroll exit debounce)
    scroll_active = False
    lmb_down = False
    # edge-trigger arming: a pinch press only fires when the pinch had
    # fully OPENED first — chatter inside the hysteresis band cannot
    # re-fire, so one pinch is always exactly one click
    armed_l = False
    lmb_pressed_at = 0.0
    engaged_at = 0.0
    last_press_t = 0.0
    last_fresh_t = time.monotonic()
    d_li_ema: float | None = None
    scroll_y_ema: float | None = None
    # median-3 windows over the raw pinch distances (see _med3)
    d48_buf: deque = deque(maxlen=3)
    d46_buf: deque = deque(maxlen=3)
    d412_buf: deque = deque(maxlen=3)
    palm_ema: float | None = None
    # per-second pinch telemetry (threshold tuning from live logs)
    win_d48 = win_d46 = win_d412 = 9.9
    win_presses = 0
    status = "raise a hand to control"
    fps_t = time.monotonic()
    fps_frames = 0
    fps = 0.0
    last_seq = -1
    canvas = None
    exit_code = 0

    def _release_buttons() -> None:
        nonlocal lmb_down
        if lmb_down:
            act.release(Button.left)
            logger.info("gesture: LMB up")
        lmb_down = False

    try:
        while True:
            frame, seq = cam.read()
            # a dead camera (unplugged mid-drag) freezes seq forever and
            # read() keeps returning the last old frame — the seq-gated
            # block below (with its grace release) then never runs again.
            # Drop held buttons on a stale freshness clock instead; a live
            # camera keeps last_fresh_t fresh (or has no buttons held).
            if (
                lmb_down
                and time.monotonic() - last_fresh_t > BUTTON_GRACE_S + 0.5
            ):
                _release_buttons()
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

                # ---- pose (fresh frames only; NO rest pose: a hand in
                # frame is active in any shape, hand-out is the pause —
                # the move/rest flicker froze the cursor 44x per session)
                pose = "none"
                raw = getattr(hands, "last_raw", None)
                if fresh and result is not None:
                    if raw is None:
                        raw = result[0]
                    gname, gscore = getattr(hands, "last_gesture", (None, 0.0))
                    official = gscore >= OFFICIAL_CONFIRM
                    if is_scroll_pose(raw) or (official and gname == "Victory"):
                        pose = "scroll"
                    else:
                        # any other hand shape drives the cursor
                        pose = "move"

                move_run = move_run + 1 if pose == "move" else 0
                idle_run = idle_run + 1 if pose == "none" else 0

                # ---- engage / disengage ----
                if (
                    not engaged
                    and result is not None
                    and move_run >= cfg.debounce_frames
                ):
                    engaged = True
                    engaged_at = now
                    # fresh gesture state on every (re)entry: the hand may
                    # return at a different distance — a stale palm EMA
                    # would mis-scale every distance for ~1s and a stale
                    # armed_l could fire a phantom click on the entry pose
                    palm_ema = None
                    d48_buf.clear()
                    d412_buf.clear()
                    d46_buf.clear()
                    armed_l = False
                    d_li_ema = None
                    # same source as the movement control point below —
                    # a shaped-vs-raw mismatch here would hop the cursor
                    anchor_pts = raw if raw is not None else result[0]
                    ax, ay = palm_center(anchor_pts)
                    act.reset(ax, ay)
                    logger.info("engaged")
                elif engaged and idle_run >= cfg.debounce_off_frames:
                    engaged = False
                    _release_buttons()
                    act.scroll_end()
                    scroll_active = False
                    logger.info("released")

                # ---- held buttons (LMB only; the RMB gesture is removed) ----
                if fresh and result is not None:
                    # slow denominator first: distances normalized by the
                    # EMA palm length stay in stable units while the hand
                    # rotates (the per-frame projected palm collapses 2-4x
                    # edge-on and used to drag every threshold along)
                    hand_len = palm_length(raw)
                    if hand_len > 1e-6:
                        palm_ema = (
                            hand_len
                            if palm_ema is None
                            else (1.0 - PALM_EMA) * palm_ema + PALM_EMA * hand_len
                        )
                    d48_buf.append(pinch_dist(raw, "index", palm_ema))
                    d412_buf.append(pinch_dist(raw, "middle", palm_ema))
                    d46_buf.append(thumb_pip_dist(raw, palm_ema))
                    d_li = _med3(d48_buf)
                    d_lm = _med3(d412_buf)
                    d46 = _med3(d46_buf)
                    win_d48 = min(win_d48, d_li)
                    win_d46 = min(win_d46, d46)
                    win_d412 = min(win_d412, d_lm)
                    d_li_ema = d_li if d_li_ema is None else (
                        PINCH_EMA * d_li + (1 - PINCH_EMA) * d_li_ema
                    )
                    # arm on a fully open pinch (edge trigger)
                    if d_li > cfg.pinch_open:
                        armed_l = True
                    if (
                        engaged
                        and not scroll_active
                        and not lmb_down
                        and now - last_press_t >= PRESS_REFRACTORY_S
                        and now - engaged_at >= ENGAGE_GRACE_S
                    ):
                        # press fires on the threshold crossing (zero
                        # added dwell); the median window is the only
                        # added latency and it kills the one-frame spikes
                        # that chained false down/up bursts
                        if armed_l and pinch_l_fire(d_li, d46, cfg.pinch_close):
                            act.press(Button.left)
                            lmb_down = True
                            armed_l = False
                            lmb_pressed_at = last_press_t = now
                            win_presses += 1
                            logger.info(
                                "gesture: LMB down d48=%.2f d46=%.2f",
                                d_li, d46,
                            )
                    # release: ONLY the pinch opening (or hand-out above /
                    # grace below). The hand may tighten freely — a natural
                    # strong pinch settles into a bundle (d46 ~0.1) and the
                    # old fist-signature drop killed every hold within
                    # 140-300ms; opening the fingers raises d48 past the
                    # open threshold and releases cleanly.
                    if lmb_down and d_li_ema > cfg.pinch_open:
                        if now - lmb_pressed_at >= MIN_HOLD_S:
                            act.release(Button.left)
                            lmb_down = False
                            logger.info("gesture: LMB up")
                elif now - last_fresh_t > BUTTON_GRACE_S:
                    _release_buttons()  # hand really gone: never leave a stuck button

                # ---- scroll ----
                scroll_run = scroll_run + 1 if pose == "scroll" else 0
                scroll_off_run = scroll_off_run + 1 if pose != "scroll" else 0
                if (
                    engaged
                    and not lmb_down
                    and scroll_run >= SCROLL_FRAMES
                ):
                    if not scroll_active:
                        scroll_active = True
                        scroll_off_run = 0
                        scroll_y_ema = raw[INDEX_MCP][1]
                        act.scroll_anchor(scroll_y_ema)
                        logger.info("gesture: scroll mode")
                elif scroll_active and scroll_off_run >= SCROLL_OFF_FRAMES:
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
                # the control point is the RAW palm center (wrist + 4
                # knuckle roots): knuckles barely move when fingers
                # pinch, and the 5-bone mean holds under wrist rotation
                # where a single knuckle slides. Inside the actuator an
                # idle freeze holds the cursor DEAD STILL whenever this
                # point is not moving (the physical-mouse contract).
                if engaged and result is not None and not scroll_active:
                    ctrl = raw if raw is not None else result[0]
                    fx, fy = palm_center(ctrl)
                    if args.dry:
                        moved += 1
                        # every frame: the old 15-frame thin-out made each
                        # call a >0.2s rebase, so the idle-freeze path was
                        # never exercised in dry mode
                        sx, sy = act.move(fx, fy, now)
                        if moved % 15 == 1:
                            logger.info("dry: cursor would move to %dx%d", sx, sy)
                    else:
                        act.move(fx, fy, now)
                        moved += 1

                if lmb_down:
                    status = "LMB (pinch index) - hold & move = drag"
                elif scroll_active:
                    status = "SCROLL - two fingers, move hand up/down"
                elif engaged:
                    status = "MOVE - hand drives  |  pinch = click  |  two fingers = scroll"
                else:
                    status = "raise a hand to control"

                frames += 1
                detected += 1 if result is not None else 0
                engaged_frames += 1 if engaged else 0
                fps_frames += 1
                if now - fps_t >= 1.0:
                    fps = fps_frames / (now - fps_t)
                    fps_frames = 0
                    fps_t = now
                    logger.info(
                        "fps %.1f  hands %s  engaged %s  lmb %s  scroll %s"
                        "  sens %.1f  pinch min d48 %.2f d46 %.2f d412 %.2f  presses %d",
                        fps,
                        bool(result),
                        engaged,
                        lmb_down,
                        scroll_active,
                        act.sensitivity,
                        win_d48,
                        win_d46,
                        win_d412,
                        win_presses,
                    )
                    win_d48 = win_d46 = win_d412 = 9.9
                    win_presses = 0

                if show_window:
                    canvas = draw(
                        frame,
                        result,
                        fps,
                        act.sensitivity,
                        status,
                        active=engaged,
                        lmb=lmb_down,
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
