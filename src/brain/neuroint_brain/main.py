"""Brain entrypoint: wire config + FSM + mapper + server."""

from __future__ import annotations

import asyncio
import logging
import sys

from neuroint_brain import __version__
from neuroint_brain.actions.mapper import load_mapping
from neuroint_brain.config import settings
from neuroint_brain.gestures.fsm import GestureFSM
from neuroint_brain.server import BrainServer


def _setup_logging() -> None:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def build() -> BrainServer:
    fsm = GestureFSM(
        arm_hold_ms=settings.arm_hold_ms,
        swipe_window_ms=settings.swipe_window_ms,
        swipe_min_distance=settings.swipe_min_distance,
        hand_lost_reset_ms=settings.hand_lost_reset_ms,
        cooldown_ms=settings.cooldown_ms,
        kill_hold_ms=settings.kill_hold_ms,
        cursor_lost_ms=settings.cursor_lost_ms,
    )
    mapper = load_mapping(settings.gestures_config)
    return BrainServer(fsm, mapper, host=settings.ws_host, port=settings.ws_port)


def main() -> None:
    _setup_logging()
    logging.getLogger(__name__).info("neuroint brain %s starting", __version__)
    server = build()
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("shutting down")


if __name__ == "__main__":
    main()
