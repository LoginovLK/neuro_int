"""Frame capture layer with swappable backends."""

from neuroint_edge.capture.base import FrameSource
from neuroint_edge.capture.factory import create_capture

__all__ = ["FrameSource", "create_capture"]
