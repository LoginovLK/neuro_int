"""Machine-local settings (config.json at the repo root).

Sensitivity is adjustable live from the debug window (+/-) and persists
here on every change. No secrets — safe for the public repo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


@dataclass
class Settings:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    sensitivity: float = 1.0
    min_cutoff: float = 1.2
    beta: float = 0.01
    palm_score_threshold: float = 0.4
    landmark_score_threshold: float = 0.25
    debounce_frames: int = 3
    debounce_off_frames: int = 10
    tracking: bool = True
    clahe: bool = True
    debug_window: bool = True

    @classmethod
    def load(cls, path: str | Path = CONFIG_PATH) -> Settings:
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                known = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in data.items() if k in known})
            except Exception as exc:  # corrupt config -> defaults
                logging.getLogger(__name__).warning("config %s ignored: %s", p, exc)
        return cls()

    def save(self, path: str | Path = CONFIG_PATH) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
