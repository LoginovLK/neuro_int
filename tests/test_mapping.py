"""Active-region mapping math."""

from neuroint_cursor.mapping import CursorMapper

SW, SH = 1920, 1080


def test_sensitivity_drives_region_size():
    m = CursorMapper(SW, SH, sensitivity=1.0)
    assert abs(m.region_fraction - 0.5) < 1e-9
    assert m.region() == (0.25, 0.25, 0.75, 0.75)

    m2 = CursorMapper(SW, SH, sensitivity=2.0)
    assert abs(m2.region_fraction - 1.0) < 1e-9

    m3 = CursorMapper(SW, SH, sensitivity=0.5)
    assert abs(m3.region_fraction - 0.25) < 1e-9


def test_region_clamped():
    assert CursorMapper(SW, SH, sensitivity=10.0).region_fraction == 1.0
    assert CursorMapper(SW, SH, sensitivity=0.01).region_fraction == 0.15


def test_region_corners_map_to_screen_corners():
    m = CursorMapper(SW, SH, sensitivity=1.0)
    assert m.map(0.25, 0.25) == (0, 0)
    assert m.map(0.75, 0.75) == (SW, SH)
    assert m.map(0.5, 0.5) == (SW // 2, SH // 2)


def test_outside_region_clamps_to_edges():
    m = CursorMapper(SW, SH, sensitivity=1.0)
    assert m.map(0.0, 0.5) == (0, SH // 2)
    assert m.map(1.0, 0.1) == (SW, 0)
