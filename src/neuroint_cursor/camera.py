"""Camera: CAP_DSHOW + MJPG + dedicated drain thread (latest frame only).

Proven facts baked in (confirmed by research):
- OpenCV CAP_DSHOW queues frames internally; a slow consumer reads STALE
  frames and lag accumulates (the old 2-3s bug). A dedicated drain thread
  that continuously reads and keeps only the newest frame removes it.
- MJPG fourcc enables 30fps at 720p (default YUY2 caps far lower on many
  webcams).
- BUFFERSIZE=1 where the backend honors it.
- read() returns (frame, seq): seq counts UNIQUE frames only — DShow
  hands out duplicates between real frames, and a strided-byte signature
  tells them apart.
- a frozen USB camera hands back the same frame at loop speed; the drain
  detects that and soft-reopens the capture.
- read() never resets to None: repeated delivery is harmless, a dropped
  frame stalls fps (old lesson).
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque

import cv2

logger = logging.getLogger(__name__)

# Fixed absolute paths, no shell: public-repo hygiene (bandit S603/S607).
_PNPUTIL = r"C:\Windows\System32\pnputil.exe"
_POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


class Camera:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
        open_retries: int = 5,
    ) -> None:
        self._mirror = mirror
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._open_retries = open_retries
        self._cap = self._open(index, width, height, fps, open_retries)
        self._reopen_times: deque[float] = deque()
        self._last_escalation = 0.0
        self._lock = threading.Lock()
        self._latest: object | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="camera-drain", daemon=True)
        self._thread.start()

    @staticmethod
    def _open(index: int, width: int, height: int, fps: int, retries: int):
        last_err: Exception = RuntimeError("camera open failed")
        for attempt in range(1, retries + 1):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
            last_err = RuntimeError(f"camera {index} not opened (attempt {attempt}/{retries})")
            time.sleep(1.0)
        raise last_err

    def _drain(self) -> None:
        prev_sig: bytes | None = None
        dup_fast = 0
        last_t = time.monotonic()
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception:  # noqa: BLE001 - driver hiccups must not kill the drain
                ok, frame = False, None
            if ok and frame is not None:
                if self._mirror:
                    frame = cv2.flip(frame, 1)
                # DShow read() hands out DUPLICATES between real frames; a
                # cheap strided signature separates them (seq must count
                # unique frames, or the loop re-runs inference on the same
                # image and the fps counter lies).
                sig = frame[::16, ::16].tobytes()
                new_frame = sig != prev_sig
                prev_sig = sig
                # A frozen camera hands back the same frame at loop speed
                # (duplicates every few ms) — soft-reopen the capture.
                now = time.monotonic()
                if not new_frame and now - last_t < 0.005:
                    dup_fast += 1
                    if dup_fast >= 20:
                        dup_fast = 0
                        self._escalate(now)
                        prev_sig = None
                else:
                    dup_fast = 0
                last_t = now
                with self._lock:
                    self._latest = frame
                    if new_frame:
                        self._seq += 1
            else:
                self._stop.wait(0.05)

    def _escalate(self, now: float) -> None:
        """Frozen camera: soft reopen; if it keeps dying, restart the USB device."""
        logger.warning("camera frozen (duplicate frames) — reopening capture")
        self._reopen()
        self._reopen_times.append(now)
        while self._reopen_times and self._reopen_times[0] < now - 30.0:
            self._reopen_times.popleft()
        if len(self._reopen_times) >= 2 and now - self._last_escalation > 60.0:
            self._last_escalation = now
            self._usb_restart()
            self._reopen()

    def _usb_restart(self) -> None:
        """PnP-restart the webcam device (the v1 un-wedge; needs elevation)."""
        dev_id = self._camera_device_id()
        if dev_id is None:
            logger.warning("usb restart: no camera device found")
            return
        logger.warning("usb restart: %s", dev_id)
        try:
            subprocess.run(  # noqa: S603 - fixed absolute binary, no shell
                [_PNPUTIL, "/restart-device", dev_id],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - watchdog must survive
            logger.error("usb restart failed: %s", exc)

    @staticmethod
    def _camera_device_id() -> str | None:
        try:
            out = subprocess.run(  # noqa: S603 - fixed absolute binary, no shell
                [
                    _POWERSHELL,
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_PnPEntity | "
                    "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
                    "Select-Object -ExpandProperty DeviceID",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - watchdog must survive
            logger.debug("camera device lookup failed: %s", exc)
            return None
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.startswith("USB\\"):
                return line
        return None

    def _reopen(self) -> None:
        logger.warning("camera frozen (duplicate frames) — reopening capture")
        try:
            self._cap.release()
        except Exception as exc:  # release must not kill the drain
            logger.debug("camera release failed: %s", exc)
        try:
            self._cap = self._open(
                self._index, self._width, self._height, self._fps, self._open_retries
            )
            logger.info("camera reopened")
        except Exception as exc:  # next reads retry via the dup watchdog
            logger.error("camera reopen failed: %s", exc)

    def read(self) -> tuple[object | None, int]:
        """Instant: returns (newest frame, its sequence number).

        The sequence lets the caller skip re-processing a frame it has
        already seen — read() repeats delivery between camera frames
        (never None once started), and duplicating inference on the same
        frame wastes GPU and lies to the fps counter.
        """
        with self._lock:
            return self._latest, self._seq

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._cap.release()
