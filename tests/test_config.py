"""Settings load/save roundtrip."""

from neuroint_cursor.config import Settings


def test_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    s = Settings(sensitivity=1.8, camera_index=1)
    s.save(p)
    s2 = Settings.load(p)
    assert s2.sensitivity == 1.8
    assert s2.camera_index == 1
    assert s2.width == 1280  # untouched default


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"sensitivity": 2.5, "legacy_gesture_field": true}', encoding="utf-8")
    s = Settings.load(p)
    assert s.sensitivity == 2.5


def test_corrupt_config_falls_back_to_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    s = Settings.load(p)
    assert s == Settings()


def test_missing_config_gives_defaults(tmp_path):
    assert Settings.load(tmp_path / "absent.json") == Settings()
