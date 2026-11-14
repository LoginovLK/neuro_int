"""WS diagnostic client: prints close code/reason and every message."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
import websockets
from neuroint_protocol.messages import HelloMsg


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.send(HelloMsg(client="diag").model_dump_json())
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", blank)
        for i in range(5):
            try:
                await ws.send(jpeg.tobytes())
            except websockets.ConnectionClosed:
                print(f"send failed at frame {i}: code={ws.close_code} reason={ws.close_reason!r}")
                return
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                kind = "binary" if isinstance(raw, bytes) else raw[:110]
                print(f"frame {i}: got {type(raw).__name__}: {kind}")
            except TimeoutError:
                print(f"frame {i}: no message in 3s")
            except websockets.ConnectionClosed:
                print(f"closed at frame {i}: code={ws.close_code} reason={ws.close_reason!r}")
                return
        print("done, connection stayed open")


asyncio.run(main())
