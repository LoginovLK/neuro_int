"""Check what the brain sees right now: hands count from StatusMsg stream."""

from __future__ import annotations

import asyncio
import time

import websockets
from neuroint_protocol.messages import StatusMsg, parse_message


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        deadline = time.monotonic() + 8.0
        seen = 0
        max_hands = 0
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
            except TimeoutError:
                break
            if isinstance(raw, bytes):
                continue
            msg = parse_message(raw)
            if isinstance(msg, StatusMsg):
                seen += 1
                max_hands = max(max_hands, msg.hands_seen)
                print(f"status: fps={msg.fps_inference} hands={msg.hands_seen} paused={msg.paused}")
    print(f"--- over {seen} statuses: max hands seen = {max_hands}")


asyncio.run(main())
