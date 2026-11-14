"""Frame enhancement for noisy low-light cameras (PS3 Eye VGA sensor).

Pipeline per frame:
  1. gray-world white balance - PS3 Eye's green/yellow cast is measured from
     channel means and neutralized automatically (gains refreshed rarely)
  2. temporal denoise         - blend with previous frame; sensor noise is
     frame-independent while real motion survives the blend
  3. CLAHE on L channel       - lifts shadow detail without blowing highlights
  4. mild unsharp mask        - restores edges after denoising

Result: landmarks survive in scenes where raw frames fall apart.
"""

from __future__ import annotations

import cv2
import numpy as np


class FrameEnhancer:
    def __init__(
        self,
        white_balance: bool = True,
        wb_refresh_interval: int = 30,
        denoise_blend: float = 0.35,
        nlm_denoise: bool = False,  # OFF by default: ~137 ms/frame on CPU, kills fps
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        sharpen_amount: float = 0.4,
    ) -> None:
        self._wb_enabled = white_balance
        self._wb_refresh_interval = max(wb_refresh_interval, 1)
        self._wb_gains: np.ndarray | None = None
        self._wb_counter = 0
        self._denoise_blend = denoise_blend
        self._nlm_denoise = nlm_denoise
        self._clahe_enabled = clahe_enabled
        self._sharpen_amount = sharpen_amount
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        self._prev: np.ndarray | None = None

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = frame_bgr

        if self._wb_enabled:
            out = self._apply_white_balance(out)

        if self._nlm_denoise:
            out = self._nlm(out)

        if self._denoise_blend > 0 and self._prev is not None:
            if self._prev.shape == out.shape:
                out = cv2.addWeighted(
                    out, 1.0 - self._denoise_blend, self._prev, self._denoise_blend, 0
                )
        self._prev = out.copy()

        if self._clahe_enabled:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lightness = lab[:, :, 0]
            lab[:, :, 0] = self._clahe.apply(lightness)
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self._sharpen_amount > 0:
            blur = cv2.GaussianBlur(out, (0, 0), 1.2)
            out = cv2.addWeighted(out, 1.0 + self._sharpen_amount, blur, -self._sharpen_amount, 0)
        return out

    def reset(self) -> None:
        self._prev = None
        self._wb_gains = None

    def _apply_white_balance(self, frame: np.ndarray) -> np.ndarray:
        self._wb_counter += 1
        if self._wb_gains is None or self._wb_counter % self._wb_refresh_interval == 0:
            means = frame.reshape(-1, 3).mean(axis=0)
            target = float(means.mean())
            if target > 1e-3:
                # Keep overall brightness; equalize channels.
                self._wb_gains = np.clip(target / (means + 1e-3), 0.0, 3.0)
        if self._wb_gains is None:
            return frame
        balanced = frame.astype(np.float32) * self._wb_gains.astype(np.float32)
        return np.clip(balanced, 0, 255).astype(np.uint8)

    def _nlm(self, frame: np.ndarray) -> np.ndarray:
        """Non-local means denoise at half scale (fast): sensor grain dies,
        hand contours survive. NLM at full 640x480 is too slow for 30fps;
        at 320x240 it costs ~20 ms and the upscale re-smooths edges."""
        small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        small = cv2.fastNlMeansDenoisingColored(small, None, 3, 3, 7, 21)
        return cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
