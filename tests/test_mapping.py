"""Full-frame -> full-screen mapping math (no active region)."""

from neuroint_cursor.mapping import CursorMapper

SW, SH = 1920, 1080


def test_frame_corners_map_to_screen_corners():
    m = CursorMapper(SW, SH, sensitivity=1.0)
    assert m.map(0.0, 0.0) == (0, 0)
    assert m.map(1.0, 1.0) == (SW - 1, SH - 1)
    assert m.map(0.5, 0.5) == (SW // 2, SH // 2)


def test_any_position_is_reachable():
    """The old active-region clamp is gone: off-center points map too."""
    m = CursorMapper(SW, SH, sensitivity=1.0)
    sx, sy = m.map(0.1, 0.2)
    assert 0 < sx < SW // 2
    assert 0 < sy < SH // 2


def test_sensitivity_is_gain_around_center():
    m1 = CursorMapper(SW, SH, sensitivity=1.0)
    m2 = CursorMapper(SW, SH, sensitivity=2.0)
    # same hand offset, double gain -> double screen offset
    x1, _ = m1.map(0.6, 0.5)
    x2, _ = m2.map(0.6, 0.5)
    assert x2 - SW // 2 == 2 * (x1 - SW // 2)


def test_gain_clamps_to_screen():
    m = CursorMapper(SW, SH, sensitivity=4.0)
    # far outside the frame with a big gain still lands on screen edges
    assert m.map(-1.0, -1.0) == (0, 0)
    assert m.map(2.0, 2.0) == (SW - 1, SH - 1)


def test_sensitivity_clamped():
    m = CursorMapper(SW, SH, sensitivity=100.0)
    assert m.map(0.6, 0.5)[0] == CursorMapper(SW, SH, sensitivity=4.0).map(0.6, 0.5)[0]
