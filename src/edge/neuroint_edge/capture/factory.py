"""Capture backend factory with graceful degradation."""

from __future__ import annotations

import logging

from neuroint_edge.capture.base import FrameSource
from neuroint_edge.config import EdgeSettings

logger = logging.getLogger(__name__)


def create_capture(settings: EdgeSettings) -> FrameSource:
    """Build the requested (or best available) capture backend.

    auto: pseyepy when installed (PS3 Eye), else plain OpenCV.
    Explicit backend selection is honored; a failing explicit pseyepy choice
    raises instead of silently falling back (surprising behavior is worse).
    """
    backend = settings.capture_backend.lower()

    if backend in ("auto", "pseyepy"):
        try:
            import pseyepy  # noqa: F401

            source = _build_pseye(settings)
            if source is not None:
                return source
        except ImportError:
            if backend == "pseyepy":
                raise
            logger.info("pseyepy not installed -> OpenCV fallback")
        except Exception as exc:
            if backend == "pseyepy":
                raise
            logger.warning("pseyepy init failed (%s) -> OpenCV fallback", exc)
    elif backend == "pseyepy":
        raise RuntimeError("unreachable")

    return _build_opencv(settings)


def _build_pseye(settings: EdgeSettings) -> FrameSource | None:
    from pseyepy import cam_count

    count = cam_count()
    if count == 0:
        logger.warning("pseyepy sees no cameras -> OpenCV fallback")
        return None
    index = min(settings.camera_index, count - 1)
    logger.info(
        "capture backend: pseyepy (index=%d, fps=%d, gain=%d, exposure=%d)",
        index,
        settings.camera_fps,
        settings.gain,
        settings.exposure,
    )
    from neuroint_edge.capture.pseye import PseyeSource

    return PseyeSource(
        index=index,
        fps=settings.camera_fps,
        gain=settings.gain,
        exposure=settings.exposure,
    )


def _build_opencv(settings: EdgeSettings) -> FrameSource:
    logger.info("capture backend: opencv (index=%d)", settings.camera_index)
    from neuroint_edge.capture.opencv import OpenCvSource

    return OpenCvSource(index=settings.camera_index, fps=settings.capture_fps)
