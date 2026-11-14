"""Smoke test: verify the running brain responds over the wire protocol.

Usage:  make smoke   (requires the stack to be up: make rebuild / make up)

Sends a hello + a burst of dummy JPEG frames and waits for a StatusMsg.
No camera, no keyboard: pure protocol verification.
"""

from __future__ import annotations

import asyncio
import os
import sys

import cv2
import numpy as np
import websockets
from neuroint_protocol.messages import HelloMsg, StatusMsg, parse_message


async def main() -> int:
    url = os.environ.get("NEUROINT_WS_URL", "ws://127.0.0.1:8765")
    print(f"smoke: connecting to {url}")
    async with websockets.connect(url) as ws:
        await ws.send(HelloMsg(client="smoke").model_dump_json())

        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", blank)
        if not ok:
            print("FAIL: could not encode dummy frame")
            return 1
        for _ in range(10):
            await ws.send(jpeg.tobytes())
            await asyncio.sleep(0.05)

        try:
            deadline = asyncio.get_event_loop().time() + 15.0
            while asyncio.get_event_loop().time() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(raw, bytes):
                    continue
                msg = parse_message(raw)
                if isinstance(msg, StatusMsg):
                    print(
                        f"OK: brain alive — fps={msg.fps_inference} "
                        f"hands={msg.hands_seen} paused={msg.paused} uptime={msg.uptime_s}s"
                    )
                    return 0
        except TimeoutError:
            print("FAIL: brain sent no status message within timeout")
            return 1
    print("FAIL: connection closed before status arrived")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
