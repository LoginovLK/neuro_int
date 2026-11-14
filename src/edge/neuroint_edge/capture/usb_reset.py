"""Windows USB device reset: un-wedge a hung camera without replugging.

When a webcam hangs at the driver level, cv2 read() times out forever and
reopening the backend fails until the USB device is restarted. pnputil can
do that from an elevated shell (the edge runs as Administrator here).
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def restart_camera_devices() -> bool:
    """Restart every present camera-class PnP device. True = something was
    restarted (best effort; requires an elevated process)."""
    try:
        # Static command + constant filter, no untrusted input (S603 safe).
        listing = subprocess.run(  # noqa: S603
            [
                _POWERSHELL,
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -PresentOnly | "
                "Where-Object { $_.Class -in 'Camera','Image' -or $_.FriendlyName -match 'camera|webcam' } | "
                "Select-Object -ExpandProperty InstanceId",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        logger.warning("camera device listing failed: %s", exc)
        return False

    restarted = False
    for instance_id in [line.strip() for line in listing.stdout.splitlines() if line.strip()]:
        try:
            # instance_id comes from Get-PnpDevice itself, never from input.
            subprocess.run(  # noqa: S603
                ["pnputil", "/restart-device", instance_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            logger.info("USB device restarted: %s", instance_id)
            restarted = True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            logger.warning("USB restart failed for %s: %s", instance_id, exc)
    return restarted
