"""Probe Lev's iteration-11 screenshots: face FP vs edge-on gun hand.

Dumps every palm box (threshold sweep), and for each box the full
landmark inference with geometry metrics — looking for the discriminator
that kills the face while keeping the real edge-on hand.
"""

from __future__ import annotations

import json
import time
from math import hypot
from pathlib import Path

import cv2

from neuroint_cursor.hands import OnnxHands
from neuroint_cursor.pose import finger_states, is_pinch, palm_ratio

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CAPS = ROOT / "captures"


def d(pts, a, b):
    return hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])


def geom(pts, frame_w=1280, frame_h=720):
    pw = d(pts, 5, 17)  # palm width (index MCP -> pinky MCP)
    lengths = {
        "index": d(pts, 5, 8),
        "middle": d(pts, 9, 12),
        "ring": d(pts, 13, 16),
        "pinky": d(pts, 17, 20),
    }
    rel = {k: round(v / pw, 2) for k, v in lengths.items()} if pw > 1e-6 else {}
    xs = [p[0] * frame_w for p in pts]
    ys = [p[1] * frame_h for p in pts]
    spread = hypot(max(xs) - min(xs), max(ys) - min(ys))
    return {
        "palm_ratio": round(palm_ratio(pts) or -1, 2),
        "finger_len_over_palmw": rel,
        "spread_px": round(spread, 1),
    }


def main() -> None:
    hands = OnnxHands(
        MODELS / "palm_detection_full_inf_post_192x192.onnx",
        MODELS / "hand_landmark_sparse_Nx3x224x224.onnx",
    )
    report = {}
    for name in ["lev_gun_edge.png", "lev_face_fp.png"]:
        img = cv2.imread(str(CAPS / name))
        h, w = img.shape[:2]
        frame = cv2.resize(img[30:h] if h > 720 else img, (1280, 720))

        # raw palm boxes at several thresholds (before cluster filter)
        boxes_thr = {}
        for thr in (0.2, 0.3, 0.4, 0.5):
            rects = hands._palm_rects(frame, thr)  # noqa: SLF001
            boxes_thr[str(thr)] = [
                [round(r[0]), round(r[1]), round(r[2]), round(r[4], 1)]
                for r in rects
            ]
        # what the pipeline sees at the adaptive thresholds
        rects = hands._cluster_filter(hands._palm_rects(frame, 0.4))  # noqa: SLF001
        cands = []
        for rect in rects:
            t0 = time.monotonic()
            out = hands._landmark_smart(frame, rect)  # noqa: SLF001
            ms = (time.monotonic() - t0) * 1000
            if out is None:
                cands.append({"rect": rect, "landmark": None, "ms": round(ms, 1)})
                continue
            pts, handed, score = out
            cands.append(
                {
                    "rect": [round(v, 1) for v in rect],
                    "landmark_score": round(score, 3),
                    "handed": handed,
                    "pointing": is_pinch(pts),
                    "states": finger_states(pts),
                    "ms": round(ms, 1),
                    **geom(pts),
                }
            )
        # timing breakdown
        t0 = time.monotonic()
        hands._palm_rects(frame, 0.4)  # noqa: SLF001
        palm_ms = (time.monotonic() - t0) * 1000
        report[name] = {
            "boxes_by_threshold": boxes_thr,
            "candidates": cands,
            "palm_stage_ms": round(palm_ms, 1),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
