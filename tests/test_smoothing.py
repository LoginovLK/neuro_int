"""One Euro Filter: noise is suppressed, fast motion still passes."""

import statistics

from neuroint_brain.vision.smoothing import OneEuroFilter


def test_static_signal_noise_reduced() -> None:
    import random

    rng = random.Random(42)  # noqa: S311 - statistical demo, not crypto
    f = OneEuroFilter(freq=30, min_cutoff=1.2, beta=0.03)
    noisy = [0.5 + rng.uniform(-0.01, 0.01) for _ in range(120)]
    filtered = [f.filter(v, i / 30.0) for i, v in enumerate(noisy)]
    assert statistics.pstdev(filtered[10:]) < statistics.pstdev(noisy[10:]) * 0.5


def test_fast_step_is_tracked() -> None:
    f = OneEuroFilter(freq=30, min_cutoff=1.2, beta=0.03)
    for i in range(30):
        f.filter(0.0, i / 30.0)
    # After an instant step the filter converges within a few frames:
    # responsive enough for swipe tracking, no second-long lag.
    out = [f.filter(1.0, (30 + i) / 30.0) for i in range(5)]
    assert out[-1] > 0.65
