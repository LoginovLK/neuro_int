"""Fine-grained: steady-state drift (no filter settling) + inference count."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from neuroint_cursor.hands import OnnxHands

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    hands = OnnxHands(
        MODELS / "palm_detection_full_inf_post_192x192.onnx",
        MODELS / "hand_landmark_sparse_Nx3x224x224.onnx",
    )
    calls = {"n": 0}
    orig = hands._landmark_from_rect

    def counted(frame, rect, min_score=None):
        calls["n"] += 1
        return orig(frame, rect, min_score)

    hands._landmark_from_rect = counted  # noqa: SLF001 - instrumenting

    report = {}
    for name in ["dbg2_TILT_1051.png", "dbg2_TILT_759.png"]:
        img = cv2.imread(str(ROOT / "captures" / name))
        h, w = img.shape[:2]
        frame = cv2.resize(img[30:h] if h > 720 else img, (1280, 720))

        hands.detect(frame)  # acquisition
        calls["n"] = 0
        tips = []
        t0 = time.monotonic()
        for _i in range(300):
            r2 = hands.detect(frame)
            if r2:
                tip = r2[0][0][8]
                tips.append((tip[0] * 1280, tip[1] * 720))
        ms = (time.monotonic() - t0) * 1000 / 300

        steady = tips[100:]
        mx = sorted(t[0] for t in steady)[len(steady) // 2]
        my = sorted(t[1] for t in steady)[len(steady) // 2]
        drift = max(abs(t[0] - mx) + abs(t[1] - my) for t in steady)
        # settle: how far sample 0 sits from steady median
        settle = abs(tips[0][0] - mx) + abs(tips[0][1] - my)
        report[name] = {
            "steady_drift_px": round(drift, 2),
            "first_sample_offset_px": round(settle, 2),
            "track_ms": round(ms, 2),
            "landmark_calls_per_300": calls["n"],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
