"""Fresh-clone bootstrap: download the runtime models into models/.

models/ is gitignored (public repo), so a new machine needs the files.
Only the PRIMARY backend files are fetched — the fallback ONNX models
are pointless when the mediapipe package (requirements.txt) is present.

Usage:  make models   (or  python tools/fetch_models.py)
"""

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

FILES = {
    # primary backend: official MediaPipe gesture recognizer bundle
    # (hand landmarks + trained gesture classes in one file)
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/latest/gesture_recognizer.task"
    ),
    # face gate (skip false hands on the face/hair)
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
}


def fetch(name: str, url: str) -> None:
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"ok   {name} ({dest.stat().st_size} bytes, cached)")
        return
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"get  {name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    # fixed URL table above, no user input -> audit noise only
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.rename(dest)
    print(f"ok   {name} ({dest.stat().st_size} bytes)")


def main() -> int:
    try:
        for name, url in FILES.items():
            fetch(name, url)
    except Exception as exc:  # noqa: BLE001 - report and fail the target
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print("models ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
