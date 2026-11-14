"""Night watchdog for the neuro_int stack.

Runs detached (pythonw) and self-heals the whole stack until the file
`.watchdog` is deleted (or the machine is shut down):

  1. brain container: docker compose up -d --wait if the WS is down
  2. edge agent:      kills all neuroint_edge pythonw processes and starts
                      a fresh one if the UI stops answering
  3. camera:          if the brain reports fps == 0 for several checks
                      (edge alive but camera wedged), restarts the USB
                      device via pnputil and then restarts the edge

Every check is appended to NIGHT_REPORT.md so the morning session (or a
compacted agent) can reconstruct the night.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "protocol"))
sys.path.insert(0, str(ROOT / "src" / "edge"))

from neuroint_protocol.messages import parse_message  # noqa: E402

CHECK_INTERVAL_S = 60
BRAIN_URL = "ws://127.0.0.1:8765"
UI_URL = "http://127.0.0.1:8756/"
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
PNPUTIL = r"C:\Windows\System32\pnputil.exe"
DOCKER = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
CAMERA_DEAD_STRIKES = 3

_report_path = ROOT / "NIGHT_REPORT.md"
_watchdog_flag = ROOT / ".watchdog"


def log(message: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with _report_path.open("a", encoding="utf-8") as report:
        report.write(f"- {line}\n")


def check_brain() -> tuple[bool, float, bool]:
    """(alive, fps, paused). fps>0 means edge + camera are producing."""

    async def probe() -> tuple[bool, float, bool, bool]:
        async with websockets.connect(BRAIN_URL, ping_interval=20, ping_timeout=15) as ws:
            t0 = time.monotonic()
            fps = 0.0
            paused = False
            hands_seen = False
            while time.monotonic() - t0 < 3.0:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                except (TimeoutError, websockets.ConnectionClosed):
                    break
                msg = parse_message(raw)
                if msg.type == "status":
                    fps, paused = msg.fps_inference, msg.paused
                    if msg.hands_seen > 0:
                        hands_seen = True
        return True, fps, paused, hands_seen

    try:
        return asyncio.run(probe())
    except Exception:  # noqa: BLE001 - any failure means brain is down
        return False, 0.0, False, False


def check_edge_ui() -> bool:
    import urllib.request

    try:
        request = urllib.request.Request(UI_URL)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=3) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def restart_brain() -> None:
    subprocess.run(  # noqa: S603, S607 - fixed absolute path, no input
        [DOCKER, "compose", "up", "-d", "--wait"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


def restart_edge() -> None:
    # Kill every edge interpreter, then start a fresh one detached.
    subprocess.run(  # noqa: S603, S607 - audited local management call
        [
            POWERSHELL,
            "-NoProfile",
            "-Command",
            'Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "pythonw.exe" '
            '-and $_.CommandLine -match "neuroint_edge" } | ForEach-Object '
            "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.Popen(  # noqa: S603, S607 - audited local management call
        [
            POWERSHELL,
            "-NoProfile",
            "-Command",
            "$p = Start-Process -WindowStyle Hidden -FilePath "
            "'.venv-edge/Scripts/pythonw.exe' -ArgumentList '-m','neuroint_edge.main' "
            "-WorkingDirectory '.' -PassThru; Set-Content -Path '.edge.pid' -Value $p.Id",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def restart_camera_usb() -> bool:
    try:
        listing = subprocess.run(  # noqa: S603, S607 - audited local management call
            [
                POWERSHELL,
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -PresentOnly | Where-Object "
                "{ $_.Class -in 'Camera','Image' } | "
                "Select-Object -ExpandProperty InstanceId",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"USB device listing failed: {exc}")
        return False
    ok = False
    for instance_id in [line.strip() for line in listing.stdout.splitlines() if line.strip()]:
        try:
            subprocess.run(  # noqa: S603 - instance_id comes from the OS device list
                [PNPUTIL, "/restart-device", instance_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            ok = True
        except Exception as exc:  # noqa: BLE001
            log(f"USB restart failed for {instance_id}: {exc}")
    return ok


def camera_strikes() -> int:
    strikes_file = ROOT / ".camera_strikes"
    try:
        return int(strikes_file.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def set_camera_strikes(n: int) -> None:
    (ROOT / ".camera_strikes").write_text(str(n), encoding="utf-8")


def main() -> int:
    log("=== night watchdog started ===")
    camera_strike_count = camera_strikes()
    while _watchdog_flag.exists():
        time.sleep(CHECK_INTERVAL_S)

        brain_alive, fps, paused, hands_seen = check_brain()
        edge_alive = check_edge_ui()

        if not brain_alive:
            log("brain WS down -> docker compose up -d --wait")
            restart_brain()
            continue

        # A pause with a hand IN frame is an intentional user pause — never
        # undo it (the user may be testing the kill switch). Only auto-clear
        # a stray pause left behind while nobody was around.
        if paused and hands_seen and edge_alive:
            log("intentional pause (hand in frame) — leaving it alone")
            continue

        if fps == 0.0 and edge_alive:
            camera_strike_count += 1
            set_camera_strikes(camera_strike_count)
            log(f"camera strike {camera_strike_count}/{CAMERA_DEAD_STRIKES} (brain fps=0)")
            if camera_strike_count >= CAMERA_DEAD_STRIKES:
                log("camera wedged -> USB device restart + edge restart")
                restart_camera_usb()
                restart_edge()
                camera_strike_count = 0
                set_camera_strikes(0)
            continue

        if not edge_alive:
            log("edge UI down -> restarting edge agent")
            restart_edge()
            continue

        if paused:
            # A pause with a hand IN frame is an intentional user pause —
            # never undo it (the user may be testing the kill switch).
            # Only a stray pause left with nobody around gets cleared.
            if hands_seen:
                log("intentional pause (hand in frame) — leaving it alone")
            else:
                log("stray pause with no hand in frame -> toggling back on")
                try:

                    async def toggle() -> None:
                        async with websockets.connect(BRAIN_URL) as ws:
                            await ws.send('{"type": "command", "action": "pause_system"}')

                    asyncio.run(toggle())
                except Exception as exc:  # noqa: BLE001
                    log(f"pause toggle failed: {exc}")
            continue

        camera_strike_count = 0
        set_camera_strikes(0)
        log(f"all green: brain fps={fps}, edge UI ok, not paused")

    log("=== watchdog stopped (flag removed) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
