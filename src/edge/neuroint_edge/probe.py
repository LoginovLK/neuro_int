"""Camera probe: diagnose capture and tune PS3 Eye parameters.

Usage:  make probe   (or python -m neuroint_edge.probe)

Saves sample shots per backend/setting into captures/ and prints metrics
(brightness, saturation drift, estimated noise). Use it to pick gain/exposure
values for .env, then rebuild nothing — edge reads .env at startup.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from neuroint_edge.capture.factory import create_capture
from neuroint_edge.capture.quality import FrameEnhancer
from neuroint_edge.config import EdgeSettings

logger = logging.getLogger(__name__)

OUT_DIR = Path("captures")


def _metrics(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Laplacian variance: low value = flat/noisy-blurry, high = crisp texture.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": sharpness,
        "green_cast": float(frame[:, :, 1].mean() - frame[:, :, 2].mean()),
    }


def _save(frame: np.ndarray, name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    cv2.imwrite(str(path), frame)
    logger.info("saved %s", path)


def run_probe() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    settings = EdgeSettings()
    enhancer = FrameEnhancer()

    combos: list[dict] = [
        {"label": "raw_defaults"},
        {"label": "manual_mid", "gain": 30, "exposure": 120},
        {"label": "manual_low_gain", "gain": 20, "exposure": 160},
    ]
    try:
        import pseyepy  # noqa: F401

        has_pseye = True
    except ImportError:
        has_pseye = False
        logger.info("pseyepy not installed — probing OpenCV backend only")

    for combo in combos:
        if has_pseye:
            source = _pseye_with(settings, combo)
        else:
            source = create_capture(settings)
        try:
            source.read()  # warm-up frame
            time.sleep(0.3)
            frame = source.read()
            if frame is None:
                logger.error("%s: no frame from %s", combo["label"], source.name)
                continue
            raw_metrics = _metrics(frame)
            enhanced = enhancer.enhance(frame)
            _save(frame, f"probe_{combo['label']}_raw")
            _save(enhanced, f"probe_{combo['label']}_enhanced")
            print(f"\n[{combo['label']}] backend={source.name}")
            for key, value in raw_metrics.items():
                print(f"  {key:12s} {value:8.2f}")
        finally:
            source.close()

    print("\nGuidelines: brightness 90-140 is comfortable; green_cast near 0 is good;")
    print("sharpness higher = crisper. Compare *_raw vs *_enhanced shots in captures/.")


def _pseye_with(settings: EdgeSettings, combo: dict):
    from neuroint_edge.capture.pseye import PseyeSource

    return PseyeSource(
        index=settings.camera_index,
        fps=60,
        gain=combo.get("gain", settings.gain),
        exposure=combo.get("exposure", settings.exposure),
    )


if __name__ == "__main__":
    run_probe()
