"""Edge runtime configuration (env-driven; reads the repo root .env)."""

from __future__ import annotations

import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict


class EdgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEUROINT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ws_url: str = "ws://127.0.0.1:8765"
    log_level: str = "INFO"

    # Local web UI (camera + bones overlay + gesture editor).
    ui_enabled: bool = True
    ui_port: int = 8756

    # gestures.yaml on the host: read/written by the UI editor.
    gestures_config: str = "configs/gestures.yaml"

    capture_backend: str = "auto"  # auto | pseyepy | opencv
    capture_fps: int = 30
    # Camera sensor fps. PS3 Eye: 30fps VGA allows a twice-longer exposure
    # than 60fps -> brighter, less noisy frames in a dark room.
    camera_fps: int = 30
    jpeg_quality: int = 80
    camera_index: int = 0

    # PS3 Eye manual tuning; -1 keeps library defaults. Auto mode on this
    # camera produces heavy green-tinted gain noise, hence manual control.
    # Pick values interactively: make tune
    gain: int = -1
    exposure: int = -1

    # Zero-touch mode: when gain/exposure are not pinned (-1), the edge walks
    # the sensor to target brightness at startup and trims it continuously.
    auto_exposure: bool = True

    # GPU hand detection (ONNX, DirectML on the RTX GPU). The PINTO models
    # are the MediaPipe hand pipeline as ONNX graphs; both are fetched by
    # make install-edge into models/.
    vision_onnx_palm: str = "models/palm_detection_full_inf_post_192x192.onnx"
    vision_onnx_landmark: str = "models/hand_landmark_sparse_Nx3x224x224.onnx"

    # Detection thresholds. Low on purpose: a hand far from the camera makes
    # a small palm with a weaker score, and a missed hand kills cursor
    # steering entirely. False positives are harmless here (landmark stage
    # re-checks every candidate with its own score).
    vision_score_threshold: float = 0.3
    vision_class_threshold: float = 0.25

    # Mirror the frame (selfie view): swipes then map intuitively —
    # swipe right -> fingertip moves right on screen.
    mirror: bool = True

    # Frame enhancement for noisy/low-light sources.
    # NLM denoise is OFF by default: fastNlMeans costs ~137 ms/frame on CPU
    # and collapses the fps; temporal blend + CLAHE keep the hand detectable.
    # Enable when needed via NEUROINT_ENHANCE_NLM=1.
    enhance: bool = True
    enhance_nlm: bool = False
    enhance_denoise: float = 0.35  # 0..1 blend of previous frame
    enhance_clahe: bool = True
    enhance_clip_limit: float = 2.0

    # Cursor steering: relative mode moves the mouse by fingertip deltas
    # (hand stays near the frame center, sensitivity controls speed);
    # absolute mode maps the fingertip position to the screen directly.
    cursor_mode: str = "relative"  # relative | absolute
    cursor_sensitivity: float = 2.0


def persist_env(key: str, value: str) -> None:
    """Persist one setting into the repo-root .env (created if missing)."""
    path = pathlib.Path(".env")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


settings = EdgeSettings()
