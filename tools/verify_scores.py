"""Per-iteration: rect geometry + landmark score on the tracking feedback loop."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from neuroint_cursor.hands import OnnxHands
from neuroint_cursor.pose import palm_ratio
from neuroint_cursor.tracking import rect_from_landmarks

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    hands = OnnxHands(
        MODELS / "palm_detection_full_inf_post_192x192.onnx",
        MODELS / "hand_landmark_sparse_Nx3x224x224.onnx",
    )
    report = {}
    for name in ["dbg2_TILT_1051.png", "dbg2_TILT_759.png"]:
        img = cv2.imread(str(ROOT / "captures" / name))
        h, w = img.shape[:2]
        frame = cv2.resize(img[30:h] if h > 720 else img, (1280, 720))
        res = hands.detect(frame)
        if not res:
            report[name] = "no acquisition"
            continue
        pts = res[0][0]
        steps = []
        for i in range(10):
            rect = rect_from_landmarks(pts, 1280, 720)
            if rect is None or rect[2] < 8:
                steps.append({"i": i, "degenerate_rect": None if rect is None else rect})
                break
            try:
                out = hands._landmark_from_rect(frame, rect, min_score=0.0)  # noqa: SLF001
            except cv2.error as exc:
                steps.append({"i": i, "cv_error": str(exc)[:80], "rect": rect})
                break
            if out is None:
                steps.append({"i": i, "score": None, "rect": rect})
                break
            pts, _h, score = out
            steps.append(
                {
                    "i": i,
                    "score": round(score, 3),
                    "palm_ratio": round(palm_ratio(pts) or -1, 2),
                    "rect": [int(v) for v in rect],
                    "spread": round(
                        (max(p[0] for p in pts) - min(p[0] for p in pts)) * 1280, 1
                    ),
                }
            )
        report[name] = steps
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
