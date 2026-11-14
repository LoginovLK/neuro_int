"""Brain runtime configuration (env-driven, 12-factor style)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class BrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEUROINT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Inside the container; compose publishes it on 127.0.0.1 only.
    ws_host: str = "0.0.0.0"  # noqa: S104
    ws_port: int = 8765
    log_level: str = "INFO"

    # FSM tuning (milliseconds / hand-relative distances).
    arm_hold_ms: int = 220
    swipe_window_ms: int = 600
    swipe_min_distance: float = 2.2  # in palm widths
    hand_lost_reset_ms: int = 350
    cooldown_ms: int = 650
    kill_hold_ms: int = 1100

    # Cursor mode: tolerate a lost hand for this long before exiting
    # (fast movements blur the frame for a few frames — keep steering).
    cursor_lost_ms: int = 600

    # Where the gestures.yaml mapping lives inside the container.
    gestures_config: Path = Path("/app/configs/gestures.yaml")


settings = BrainSettings()
