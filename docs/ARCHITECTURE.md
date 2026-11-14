# Architecture

## Why edge/gateway

Docker Desktop on Windows cannot (a) reach a libusb camera attached to the
host and (b) inject keyboard events into the host's GUI session. The only
honest split:

- **brain** (container): everything that is pure computation — landmark
  inference, gesture recognition, action mapping, protocol. `make rebuild`
  applies every change to this logic in one shot.
- **edge** (host agent): thin I/O bridges — camera capture and keyboard
  actuation. It changes rarely and contains no recognition logic.

## Packages

```
src/
├── protocol/          # shared wire contract (pydantic models), no I/O
│   └── neuroint_protocol/   gestures.py, messages.py
├── brain/             # containerized inference service
│   ├── Dockerfile
│   └── neuroint_brain/
│       ├── config.py        # env-driven settings
│       ├── vision/          # landmarker.py (MediaPipe), hands.py, smoothing.py
│       ├── gestures/        # poses.py (static), fsm.py (dynamic)
│       ├── actions/         # registry.py (whitelist), mapper.py (yaml)
│       ├── server/          # app.py (WebSocket service)
│       └── main.py
└── edge/              # host agent
    └── neuroint_edge/
        ├── config.py
        ├── capture/         # pseye.py, opencv.py, quality.py, factory.py
        ├── actuator.py      # pynput, re-validates every command
        ├── client.py        # WS client loops (capture / commands)
        ├── probe.py         # camera diagnostics
        └── main.py
```

No god files: every module has one responsibility, dependencies flow
downward (`protocol ← brain, edge`; brain and edge never import each other).

## Wire protocol

Single WebSocket (loopback only, `127.0.0.1:8765`):

- **binary frames edge → brain**: JPEG, no envelope
- **text messages**: JSON models from `neuroint_protocol.messages`
  (`hello`, `status`, `gesture_event`, `command`, `error`)

Backpressure: while the inference executor is busy, newly arriving frames are
dropped. Real-time control wants the newest frame, never a stale queue.

## Gesture pipeline

```
frame → MediaPipe HandLandmarker (VIDEO mode)
      → HandObservation (21 normalized landmarks, palm_size scale unit)
      → classify_pose()            (index_up / two_up / open_palm / fist)
      → One Euro Filter on index tip
      → GestureFSM                 (IDLE → ARMED → SWIPE/HOLD → COOLDOWN)
      → ActionMapper (gestures.yaml against action whitelist)
      → CommandMsg → edge → pynput
```

FSM state machine (`gestures/fsm.py`):

| Transition | Condition |
|---|---|
| IDLE → ARMED | pose held ≥ `arm_hold_ms` (default 220) |
| ARMED → emit swipe | tip travels ≥ `swipe_min_distance` (2.2 palm widths) within `swipe_window_ms` (600) |
| ARMED → emit hold | hold pose kept ≥ `kill_hold_ms` (1100) |
| ARMED → IDLE | pose changed or hand lost > `hand_lost_reset_ms` (350) |
| after any emit | COOLDOWN 650 ms |

Distances are measured in **palm widths** (`wrist→middle_mcp`), so thresholds
are distance-invariant: sitting closer or further does not change behavior.

Swipe direction requires one axis to dominate (`|dx| > 1.6·|dy|`): diagonal
noise does not fire accidental events. Edge mirrors the frame (`NEUROINT_MIRROR=1`)
so "swipe right" means *your* right.

## Security model

See [SECURITY.md](SECURITY.md). Short version: loopback-only port, action
whitelist enforced twice (brain registry + edge key table), no dynamic code
from config, gitleaks in CI.
