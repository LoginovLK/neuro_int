"""FrameEnhancer: gray-world WB neutralizes the PS3 Eye green cast."""

import numpy as np
from neuroint_edge.capture.quality import FrameEnhancer


def _solid(b: int, g: int, r: int, size: int = 64) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = (b, g, r)
    return frame


def test_green_cast_is_neutralized():
    enhancer = FrameEnhancer(
        white_balance=True, denoise_blend=0, clahe_enabled=False, sharpen_amount=0
    )
    # Strong green-dominant frame (the complained-about PS3 Eye look).
    frame = _solid(40, 120, 45)
    out = enhancer.enhance(frame)
    means = out.reshape(-1, 3).mean(axis=0)
    spread = float(means.max() - means.min())
    assert spread < 25  # channels pulled together


def test_wb_disabled_keeps_cast():
    enhancer = FrameEnhancer(
        white_balance=False, denoise_blend=0, clahe_enabled=False, sharpen_amount=0
    )
    frame = _solid(40, 120, 45)
    out = enhancer.enhance(frame)
    assert int(out[:, :, 1].mean()) == 120  # green untouched


def test_nlm_reduces_noise():
    enhancer = FrameEnhancer(
        white_balance=False,
        denoise_blend=0,
        nlm_denoise=True,
        clahe_enabled=False,
        sharpen_amount=0,
    )
    rng = np.random.default_rng(7)
    clean = np.full((96, 128, 3), 110, dtype=np.uint8)
    noisy = np.clip(clean.astype(int) + rng.integers(-25, 25, clean.shape), 0, 255).astype(np.uint8)
    out = enhancer.enhance(noisy)
    assert float(out.std()) < float(noisy.std()) * 0.6  # grain noticeably reduced
