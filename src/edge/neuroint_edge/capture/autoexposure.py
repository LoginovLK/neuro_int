"""Automatic exposure/gain calibration for the PS3 Eye.

At startup the controller walks the sensor from a mid guess toward the
target brightness (exposure first, gain as backup — long exposure is free
brightness while gain is free noise). After settling it keeps a light
trim running so a lights-off moment self-corrects within seconds.

Runs only when the user has not pinned NEUROINT_GAIN/NEUROINT_EXPOSURE.
"""

from __future__ import annotations

import logging

import cv2

logger = logging.getLogger(__name__)


class AutoExposureController:
    """Adjusts a pseyepy Camera (mutable gain/exposure) toward target brightness."""

    def __init__(
        self,
        sensor,  # pseyepy Camera instance
        target: float = 110.0,
        deadband: float = 18.0,
        max_exposure: int = 200,
        min_exposure: int = 60,
        max_gain: int = 48,
        min_gain: int = 8,
        settle_frames: int = 5,
        trim_every_frames: int = 150,
    ) -> None:
        self._sensor = sensor
        self._target = target
        self._deadband = deadband
        self._max_exposure = max_exposure
        self._min_exposure = min_exposure
        self._max_gain = max_gain
        self._min_gain = min_gain
        self._settle_frames = settle_frames
        self._trim_every_frames = trim_every_frames

        self.exposure = 120
        self.gain = 30
        self._in_band_streak = 0
        self._settled = False
        self._frame_counter = 0
        self._apply()

    @property
    def settled(self) -> bool:
        return self._settled

    def step(self, frame_bgr) -> None:
        """Feed one RAW frame (before mirror/enhance: measure the sensor)."""
        self._frame_counter += 1
        if self._settled and self._frame_counter % self._trim_every_frames != 0:
            return  # trim pass only
        brightness = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())
        self._adjust(brightness)
        self._apply()

    def _adjust(self, brightness: float) -> None:
        low, high = self._target - self._deadband, self._target + self._deadband
        if low <= brightness <= high:
            self._in_band_streak += 1
            if not self._settled and self._in_band_streak >= self._settle_frames:
                self._settled = True
                logger.info(
                    "auto exposure settled: gain=%d exposure=%d (brightness %.0f)",
                    self.gain,
                    self.exposure,
                    brightness,
                )
            return
        self._in_band_streak = 0

        step = self._trim_every_frames // 15 if self._settled else 1
        if brightness < low:
            self._brighten(step)
        else:
            self._darken(step)

    def _brighten(self, step: int) -> None:
        if self.exposure < self._max_exposure:
            self.exposure = min(self.exposure + 12 * step, self._max_exposure)
        elif self.gain < self._max_gain:
            self.gain = min(self.gain + 4 * step, self._max_gain)
        else:
            self._settled = True  # sensor ceiling: nothing left to trade

    def _darken(self, step: int) -> None:
        if self.gain > self._min_gain:
            self.gain = max(self.gain - 4 * step, self._min_gain)
        elif self.exposure > self._min_exposure:
            self.exposure = max(self.exposure - 12 * step, self._min_exposure)

    def _apply(self) -> None:
        self._sensor.exposure = self.exposure
        self._sensor.gain = self.gain
