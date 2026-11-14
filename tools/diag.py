"""Deep diagnosis: per-detection geometry metrics + annotated crops.

Runs ~10s, dumps one JSON line per detection (score, rect, geometry
ratios) and saves up to 6 crops (the actual landmark-model input) plus
one annotated full frame — the evidence for threshold/geometry rules.
"""

from __future__ import annotations

import json
import time
from math import hypot
from pathlib import Path

import cv2

from neuroint_cursor.camera import Camera
from neuroint_cursor.hands import OnnxHands, _rotate_and_crop
from neuroint_cursor.pose import is_pointing

MODELS = Path(__file__).resolve().parents[1] / "models"
OUT = Path(__file__).resolve().parents[1] / "captures"


def _geom(pts):
    d = lambda a, b: hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])  # noqa: E731
    palm_w = d(5, 17)
    palm_l = d(0, 9)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    diag = hypot(max(xs) - min(xs), max(ys) - min(ys))
    return {
        "palm_l_over_w": round(palm_l / palm_w, 2) if palm_w > 1e-6 else None,
        "diag": round(diag, 3),
        "max_from_wrist": round(max(d(0, t) for t in (8, 12, 16, 20)), 3),
        "palm_w": round(palm_w, 3),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cam = Camera(mirror=True)
    hands = OnnxHands(
        MODELS / "palm_detection_full_inf_post_192x192.onnx",
        MODELS / "hand_landmark_sparse_Nx3x224x224.onnx",
    )
    t_end = time.monotonic() + 10.0
    crops = 0
    frame_saved = False
    n = 0
    while time.monotonic() < t_end:
        frame, _seq = cam.read()
        if frame is None:
            time.sleep(0.02)
            continue
        n += 1
        h, w = frame.shape[:2]
        # full pipeline bypass: raw candidates with metrics
        for rect in hands._palm_rects(frame):  # noqa: SLF001
            out = hands._landmark_from_rect(frame, rect)  # noqa: SLF001
            if out is None:
                print(json.dumps({"kind": "palm_only", "rect": [round(v) for v in rect]}))
                continue
            pts, handed, score = out
            rec = {
                "kind": "det",
                "score": round(score, 3),
                "pointing": is_pointing(pts),
                "rect": [round(v) for v in rect],
                **_geom(pts),
            }
            print(json.dumps(rec))
            if crops < 6:
                crops += 1
                crop = _rotate_and_crop(frame, np_float(rect))
                cv2.imwrite(str(OUT / f"diag_crop_{crops}.png"), crop)
        if not frame_saved:
            frame_saved = True
            cv2.imwrite(str(OUT / "diag_frame.png"), frame)
        time.sleep(0.05)
    cam.close()
    print(json.dumps({"frames_seen": n, "crops_saved": crops}))


def np_float(rect):

    return __import__("numpy").asarray(rect, dtype="float32")


if __name__ == "__main__":
    main()
