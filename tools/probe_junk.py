"""Why didn't the static quarantine fire on Lev's junk region?

Simulates 8 seconds (240 frames) of detect() on the still junk frame and
logs: palm boxes, winner rect trajectory, landmark score, pointing
states, quarantine events. The static check quarantines a still,
never-pointing object after 2s — this probe shows which condition fails.
"""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path

import cv2

from neuroint_cursor.hands import OnnxHands
from neuroint_cursor.pose import finger_states, is_pinch, palm_ratio

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    hands = OnnxHands(
        MODELS / "palm_detection_full_inf_post_192x192.onnx",
        MODELS / "hand_landmark_sparse_Nx3x224x224.onnx",
    )
    img = cv2.imread(str(ROOT / "captures" / "lev_junk_fp.png"))
    h, w = img.shape[:2]
    frame = cv2.resize(img[30:h] if h > 720 else img, (1280, 720))

    centers = []
    log = []
    pointing_frames = 0
    detected = 0
    for i in range(240):
        res = hands.detect(frame)
        if res:
            detected += 1
            pts = res[0][0]
            if is_pinch(pts):
                pointing_frames += 1
            # reconstruct the rect the pipeline used (prev_rect)
            centers.append(hands._prev_rect[:2] if hands._prev_rect else None)  # noqa: SLF001
            if i % 30 == 0:
                log.append(
                    {
                        "i": i,
                        "score": round(res[0][2], 2),
                        "states": finger_states(pts),
                        "pointing": is_pinch(pts),
                        "ratio": round(palm_ratio(pts) or -1, 2),
                        "rect": [round(v) for v in (hands._prev_rect or [])],  # noqa: SLF001
                    }
                )
        else:
            centers.append(None)
            if i % 30 == 0:
                log.append({"i": i, "no_detection": True})

    valid = [c for c in centers if c]
    spread = 0.0
    if len(valid) > 10:
        xs = [c[0] for c in valid]
        ys = [c[1] for c in valid]
        spread = hypot(max(xs) - min(xs), max(ys) - min(ys))

    print(
        json.dumps(
            {
                "detected_frames": detected,
                "pointing_frames": pointing_frames,
                "rect_center_spread_px": round(spread, 1),
                "quarantine_threshold_px": round(0.10 * 720, 1),
                "log": log,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
