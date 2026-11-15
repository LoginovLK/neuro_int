"""Offline probe: official canned gestures on Lev's saved capture frames.

Incident diagnostics (not a test): shows what the GestureRecognizer maps
each known frame to, what the geometric fallback says, and whether the
strict is_zero (with the new reach gate) can false-fire on the bundle.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neuroint_cursor.mp_hands import MediaPipeHands  # noqa: E402
from neuroint_cursor.pose import classify, is_zero  # noqa: E402

GESTURE_MAP = {
    "Closed_Fist": "control",
    "Pointing_Up": "point1",
    "Victory": "point2",
    "Open_Palm": "open",
}
GESTURE_MIN_SCORE = 0.55

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
    res = hands.detect(img)
    gname, gscore = hands.last_gesture
    mapped = GESTURE_MAP.get(gname) if gscore >= GESTURE_MIN_SCORE else None
    pts = res[0][0] if res else None
    geo = classify(pts) if pts else "none"
    zero = is_zero(pts) if pts else False
    print(
        f"{name}: detect={bool(res)} "
        f"official={gname}({gscore:.2f}) mapped={mapped} "
        f"geo_fallback={geo} is_zero={zero}"
    )
