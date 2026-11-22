"""Offline probe: gesture vocabulary on Lev's saved capture frames.

Incident diagnostics (not a test): shows pinch distances and pose classes
per known frame so thresholds can be tuned without a live camera.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neuroint_cursor.mp_hands import MediaPipeHands  # noqa: E402
from neuroint_cursor.pose import (  # noqa: E402
    classify,
    is_rest_pose,
    is_scroll_pose,
    pinch_dist,
    pinch_l_valid,
    thumb_pip_dist,
)

CLOSE = 0.32

root = Path(__file__).resolve().parents[1]
hands = MediaPipeHands(root / "models" / "gesture_recognizer.task")

names = [
    "lev_bundled",
    "lev_pinch_v2",
    "lev_open_palm",
    "lev_gun_edge",
    "lev_face_fp",
    "lev_junk_fp",
    "lev_hair_fp",
]
for name in names:
    img = cv2.imread(str(root / "captures" / f"{name}.png"))
    if img is None:
        print(f"{name}: READ FAIL")
        continue
    # still images run inside the coast window: without this the probe
    # would print the PREVIOUS image's measurements on a miss
    hands._coast = None
    hands._sel_wrist = None
    res = hands.detect(img)
    gname, gscore = hands.last_gesture
    pts = getattr(hands, "last_raw", None) or (res[0][0] if res else None)
    if pts is None:
        print(f"{name}: no hand")
        continue
    print(
        f"{name}: official={gname}({gscore:.2f}) class={classify(pts)} "
        f"rest={is_rest_pose(pts)} scroll={is_scroll_pose(pts)} "
        f"d_thumb_index={pinch_dist(pts, 'index'):.2f} "
        f"d_thumb_pip={thumb_pip_dist(pts):.2f} "
        f"d_thumb_middle={pinch_dist(pts, 'middle'):.2f} "
        f"lmb={pinch_l_valid(pts, CLOSE)}"
    )
