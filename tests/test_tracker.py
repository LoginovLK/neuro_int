"""Template tracker: translation measurement and response drop."""

import numpy as np

from neuroint_cursor.tracker import HandTemplateTracker

TEX = np.random.RandomState(3).rand(400, 400).astype(np.float32)
FRAME = 240
CENTER = (120, 120)
HALF = 40
BASE = 100  # keeps every window inside the texture


def _crop(tex, dx, dy):
    """Frame whose content is shifted by (+dx, +dy) vs _crop(tex, 0, 0)."""
    return tex[BASE - dy : BASE - dy + FRAME, BASE - dx : BASE - dx + FRAME]


def _points():
    return [(0.5 + i * 0.001, 0.5 - i * 0.002) for i in range(21)]


def test_translation_measured():
    t = HandTemplateTracker()
    t.capture(_crop(TEX, 0, 0), CENTER, 120.0, _points(), "Right", 0.8)
    dx, dy = 13, -9
    res = t.update(_crop(TEX, dx, dy))
    assert res is not None
    points, handed, score, center = res
    assert abs(center[0] - (CENTER[0] + dx)) < 1.0
    assert abs(center[1] - (CENTER[1] + dy)) < 1.0
    # landmarks translated by the same shift
    p0 = _points()[0]
    assert abs(points[0][0] - (p0[0] + dx / FRAME)) < 0.01
    assert abs(points[0][1] - (p0[1] + dy / FRAME)) < 0.01
    assert score == 0.8 and handed == "Right"


def test_texture_change_drops_tracker():
    t = HandTemplateTracker()
    t.capture(_crop(TEX, 0, 0), CENTER, 120.0, _points(), "Right", 0.8)
    noise = np.random.RandomState(99).rand(FRAME, FRAME).astype(np.float32)
    assert t.update(noise) is None


def test_edge_exit_drops_tracker():
    t = HandTemplateTracker()
    t.capture(_crop(TEX, 0, 0), CENTER, 120.0, _points(), "Right", 0.8)
    # shift far enough that the patch would leave the frame
    res = t.update(_crop(TEX, FRAME - 2 * HALF + 10, 0))
    assert res is None


def test_age_increments():
    t = HandTemplateTracker()
    t.capture(_crop(TEX, 0, 0), CENTER, 120.0, _points(), "Right", 0.8)
    t.update(_crop(TEX, 1, 0))
    t.update(_crop(TEX, 2, 0))
    assert t.age == 2
