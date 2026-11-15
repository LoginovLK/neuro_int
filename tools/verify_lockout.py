"""Sticker-lockout verification (iteration 12).

The live failure: a static hallucination reads pointing forever, engages
the cursor and sits there. Expected now:
  phase 1 — static pointing hand: detections stop after ~4-5 REAL seconds
            (sticker lock; offline loop runs faster than real time, so
            frame counts differ from live 30fps),
  phase 2 — same frame: stays locked (no re-flash cycle),
  phase 3 — hand shifted 160px: lock opens, detection resumes.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from neuroint_cursor.hands import OnnxHands
from neuroint_cursor.pose import is_pinch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    hands = OnnxHands(
        MODELS / "palm_detection_full_inf_post_192x192.onnx",
        MODELS / "hand_landmark_sparse_Nx3x224x224.onnx",
    )
    img = cv2.imread(str(ROOT / "captures" / "lev_gun_edge.png"))
    h, w = img.shape[:2]
    frame = cv2.resize(img[30:h] if h > 720 else img, (1280, 720))
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 160], [0, 1, 0]]), (1280, 720))

    report = {}

    def run(frames, f, label):
        last_det = None
        dets = 0
        for _ in range(frames):
            r = hands.detect(f)
            if r:
                dets += 1
                last_det = is_pinch(r[0][0])
        report[label] = {
            "detections": dets,
            "of": frames,
            "last_pointing": last_det,
            "locked": hands._lock is not None,  # noqa: SLF001
        }

    run(700, frame, "phase1_static_pointing")
    run(150, frame, "phase2_still_locked")
    run(250, shifted, "phase3_hand_moved_unlock")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
