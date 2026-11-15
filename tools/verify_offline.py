"""Offline verification of the MediaPipe-faithful pipeline (no camera).

Takes saved debug frames, runs acquisition, then 300 tracking frames on
the same image, and reports: score, pose, per-frame timings and BONE
DRIFT (how far the fingertip wanders across 300 frames — the "кости
едут" regression). On a static hand MediaPipe tracking is idempotent:
the rect is derived from stable landmarks, so drift must be ~0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from neuroint_cursor.hands import OnnxHands
from neuroint_cursor.pose import is_pinch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CAPS = ROOT / "captures"
FRAMES = ["dbg2_TILT_1051.png", "dbg2_TILT_759.png"]


def load_frame(path: Path) -> cv2.Mat | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if h > 720:  # cut the debug-window title bar on screenshots
        img = img[30:h]
    return cv2.resize(img, (1280, 720))


def main() -> None:
    hands = OnnxHands(
        MODELS / "palm_detection_full_inf_post_192x192.onnx",
        MODELS / "hand_landmark_sparse_Nx3x224x224.onnx",
    )
    report: dict[str, object] = {}
    for name in FRAMES:
        frame = load_frame(CAPS / name)
        if frame is None:
            report[name] = "not found"
            continue

        t0 = time.monotonic()
        res = hands.detect(frame)
        acq_ms = (time.monotonic() - t0) * 1000
        rec: dict[str, object] = {"acq_ms": round(acq_ms, 1)}
        if not res:
            report[name] = {**rec, "detected": False}
            continue

        pts, _handed, score = res[0]
        tip0 = (pts[8][0] * 1280, pts[8][1] * 720)
        rec.update(
            detected=True,
            score=round(score, 3),
            pointing=is_pinch(pts),
            tip0=[round(tip0[0], 1), round(tip0[1], 1)],
        )

        tips: list[tuple[float, float]] = [tip0]
        t0 = time.monotonic()
        for i in range(300):
            r2 = hands.detect(frame)
            if i % 50 == 0 and r2:
                tip = r2[0][0][8]
                tips.append((tip[0] * 1280, tip[1] * 720))
        track_ms = (time.monotonic() - t0) * 1000 / 300

        drift = 0.0
        if tips:
            mx = sorted(t[0] for t in tips)[len(tips) // 2]
            my = sorted(t[1] for t in tips)[len(tips) // 2]
            drift = max(abs(t[0] - mx) + abs(t[1] - my) for t in tips)
        rec.update(
            track_ms=round(track_ms, 2),
            drift_px=round(drift, 2),
            track_samples=len(tips),
        )
        report[name] = rec

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
