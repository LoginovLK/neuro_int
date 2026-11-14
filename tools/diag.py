"""Capture one annotated detection frame for offline diagnosis.

Point your hand at the camera while this runs (~8s). Saves
captures/diag_hands.png with the best-scoring detection and prints its
stats as JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from neuroint_cursor.camera import Camera
from neuroint_cursor.hands import OnnxHands
from neuroint_cursor.overlay import draw
from neuroint_cursor.pose import finger_states, is_pointing

MODELS = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "captures"
    out_dir.mkdir(exist_ok=True)
    cam = Camera(mirror=True)
    hands = OnnxHands(
        MODELS / "palm_detection_full_inf_post_192x192.onnx",
        MODELS / "hand_landmark_sparse_Nx3x224x224.onnx",
    )
    best = None
    n = 0
    detections = 0
    t_end = time.monotonic() + 8.0
    while time.monotonic() < t_end:
        frame, _seq = cam.read()
        if frame is None:
            time.sleep(0.02)
            continue
        n += 1
        results = hands.detect(frame)
        if results:
            detections += 1
            if best is None or results[0][2] > best[1][2]:
                best = (frame.copy(), results[0])
        time.sleep(0.01)
    cam.close()

    if best is None:
        print(json.dumps({"frames": n, "detections": 0}))
        return
    frame, (pts, hand, score) = best
    canvas = draw(frame, (pts, hand, score), (0.25, 0.25, 0.75, 0.75), 30, 1.0, is_pointing(pts))
    out = out_dir / "diag_hands.png"
    cv2.imwrite(str(out), cv2.resize(canvas, (960, 540)))
    print(
        json.dumps(
            {
                "frames": n,
                "detections": detections,
                "best_score": round(score, 3),
                "fingers": finger_states(pts),
                "pointing": is_pointing(pts),
                "tip8": [round(v, 3) for v in pts[8]],
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
