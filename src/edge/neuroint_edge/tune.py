"""Interactive camera tuner: live preview, keyboard gain/exposure control.

Usage:  make tune   (stop the edge agent first: the camera is exclusive)

Keys:
  up / down     - gain -/+ 1         (higher = brighter, more noise)
  left / right  - exposure -/+ 5     (higher = brighter, more motion blur)
  e             - toggle frame enhancement (denoise + CLAHE)
  m             - toggle mirror (the edge mirrors frames by default)
  s             - print the .env snippet for the current values
  esc / q       - quit (snippet printed at exit as well)

Windows-only preview (cv2.imshow).
"""

from __future__ import annotations

import sys

import cv2

from neuroint_edge.capture.quality import FrameEnhancer
from neuroint_edge.config import settings

GAIN_MIN, GAIN_MAX = 0, 63  # PS3 Eye sensor range
EXP_MIN, EXP_MAX = 0, 255

# cv2.waitKeyEx arrow key codes on Windows.
KEY_UP, KEY_DOWN = 2490368, 2621440
KEY_LEFT, KEY_RIGHT = 2424832, 2555904


def main() -> int:
    try:
        import pseyepy  # noqa: F401
    except ImportError:
        print("pseyepy is not installed: run make install-edge")
        return 1

    from neuroint_edge.capture.pseye import PseyeSource

    try:
        source = PseyeSource(index=settings.camera_index, fps=settings.camera_fps)
    except Exception as exc:  # noqa: BLE001 - any open failure means busy/absent
        print(f"camera unavailable: {exc}")
        print("hint: the edge agent holds the camera -> run: make edge-stop")
        return 1

    gain = settings.gain if settings.gain >= 0 else 30
    exposure = settings.exposure if settings.exposure >= 0 else 120
    enhance = True
    mirror = settings.mirror
    enhancer = FrameEnhancer()
    source.sensor.gain = gain
    source.sensor.exposure = exposure

    print(
        "keys: up/down gain, left/right exposure, e=enhance, m=mirror, s=.env snippet, q/esc=quit"
    )
    window = "neuro_int camera tuner"
    try:
        while True:
            frame = source.read()
            if frame is None:
                if cv2.waitKey(50) != -1:
                    continue
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            shown = enhancer.enhance(frame) if enhance else frame

            hud = (
                f"gain {gain:2d}   exposure {exposure:3d}   "
                f"enhance [{'on' if enhance else 'off'}]   mirror [{'on' if mirror else 'off'}]"
            )
            cv2.rectangle(shown, (0, 0), (640, 42), (10, 10, 10), -1)
            cv2.putText(
                shown,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, shown)

            key = cv2.waitKeyEx(30)
            if key == -1:
                continue
            key &= 0xFFFF
            if key == ord("q") or key == 27:
                break
            if key == ord("e"):
                enhance = not enhance
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("s"):
                _print_env_snippet(gain, exposure)
            elif key == KEY_UP:
                gain = min(gain + 1, GAIN_MAX)
                source.sensor.gain = gain
            elif key == KEY_DOWN:
                gain = max(gain - 1, GAIN_MIN)
                source.sensor.gain = gain
            elif key == KEY_RIGHT:
                exposure = min(exposure + 5, EXP_MAX)
                source.sensor.exposure = exposure
            elif key == KEY_LEFT:
                exposure = max(exposure - 5, EXP_MIN)
                source.sensor.exposure = exposure
    finally:
        source.close()
        cv2.destroyAllWindows()

    _print_env_snippet(gain, exposure)
    return 0


def _print_env_snippet(gain: int, exposure: int) -> None:
    print("\n# --- paste into .env, then: make restart-edge ---")
    print(f"NEUROINT_GAIN={gain}")
    print(f"NEUROINT_EXPOSURE={exposure}")
    print("# ------------------------------------------------")


if __name__ == "__main__":
    sys.exit(main())
