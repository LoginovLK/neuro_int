"""AutoExposureController: walks gain/exposure to target brightness."""

import cv2
import numpy as np
from neuroint_edge.capture.autoexposure import AutoExposureController


class FakeSensor:
    """Stand-in for a pseyepy Camera: brightens with exposure and gain."""

    def __init__(self, base_brightness: float = 20.0) -> None:
        self.exposure = 120
        self.gain = 30
        self._base = base_brightness

    def frame(self) -> np.ndarray:
        brightness = min(self._base + self.exposure * 0.35 + self.gain * 0.9, 255.0)
        return np.full((240, 320, 3), brightness, dtype=np.uint8)


def test_settles_on_dark_scene():
    sensor = FakeSensor(base_brightness=10.0)
    ctrl = AutoExposureController(sensor)
    for _ in range(400):
        ctrl.step(sensor.frame())
        if ctrl.settled:
            break
    assert ctrl.settled
    brightness = float(cv2.cvtColor(sensor.frame(), cv2.COLOR_BGR2GRAY).mean())
    assert 90 <= brightness <= 130


def test_darken_when_too_bright():
    sensor = FakeSensor(base_brightness=60.0)
    sensor.exposure = 200
    sensor.gain = 48
    ctrl = AutoExposureController(sensor)
    for _ in range(400):
        ctrl.step(sensor.frame())
        if ctrl.settled:
            break
    assert ctrl.exposure < 200 or ctrl.gain < 48


def test_trim_keeps_brightness_after_light_change():
    sensor = FakeSensor(base_brightness=10.0)
    ctrl = AutoExposureController(sensor)
    for _ in range(400):
        ctrl.step(sensor.frame())
        if ctrl.settled:
            break
    assert ctrl.settled
    # Lights get dimmer: sensor model base drops hard.
    sensor._base = -40.0  # noqa: SLF001 - test fixture
    for _ in range(2000):
        ctrl.step(sensor.frame())
    brightness = float(cv2.cvtColor(sensor.frame(), cv2.COLOR_BGR2GRAY).mean())
    assert brightness > 60  # recovered noticeably toward the target
