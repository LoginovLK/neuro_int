"""Frame source interface shared by capture backends."""

from __future__ import annotations

import typing

import numpy as np


class FrameSource(typing.Protocol):
    """A live camera producing BGR frames."""

    name: str

    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...
