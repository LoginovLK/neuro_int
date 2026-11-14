# neuro_int

Gesture-controlled system automation through a camera. Show a pose, swipe,
the machine obeys. v1 runs on a **PS3 Eye** camera (libusb) with a graceful
fallback to any regular webcam.

```
┌─────────────── host (Windows) ───────────────┐   ┌─── docker ───┐
│  PS3 Eye ──> edge agent ──JPEG frames──>  brain: MediaPipe HandLandmarker
│              (capture +    │                │        ↓ gesture FSM
│               quality)     │                │        ↓ action mapping
│  keyboard <── edge agent <──commands──      │        ↓ WebSocket    │
└──────────────────────────────────────────────┘   └───────────────┘
```

Brain (inference + gesture logic) lives in a container; the edge agent on the
host owns the camera and the keyboard — Docker cannot reach either.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Gestures

| Gesture | Meaning |
|---|---|
| ☝ hold + swipe right/left | Alt+Tab / Alt+Shift+Tab |
| ✌ hold + swipe left/right | Previous / next virtual desktop |
| 🖐 hold ~1s | Pause / resume gesture control (kill switch) |
| 🤏 pinch (hold) | **Cursor mode**: thumb+index steer the mouse; quick release-regrip = left click; open palm exits |

Full dictionary, cursor mode details and extension guide:
[docs/GESTURES.md](docs/GESTURES.md).

## Live UI

`make run` starts a local dashboard (loopback only):

```
make run            # edge agent: camera -> brain -> keyboard/mouse
make ui             # open http://127.0.0.1:8756 in the browser
```

- **live video with bones**: 21 landmarks + skeleton over the camera feed,
  current pose, FSM state, cursor steering point
- **gesture map**: every configured gesture as a card (pose emoji, motion,
  action, key chips) — with one-click remove
- **add rules**: pick pose + motion + action, press *Add rule* — applied
  instantly (no rebuild)
- **custom combos**: *Record key combo* captures any shortcut with a global
  keyboard hook (Alt+Tab included) and creates a reusable `custom_N` action

## Quickstart

Prerequisites: Windows 10/11, Docker Desktop, Python 3.12, PS3 Eye plugged in
with a libusb driver (Zadig) — verify with `Get-CimInstance Win32_PnPEntity |
Where-Object Name -match 'PS3Eye'`.

```bash
# 1. Host agent (edge): venv + packages, incl. pseyepy compiled from source.
#    Requires MSVC build tools (VS 2022 Community is fine).
make install-edge

# 2. Container stack (brain): build, run, wait for health.
make rebuild

# 3. Protocol smoke test (no camera needed).
make smoke

# 4. Camera probe: saves sample shots to captures/, prints tuning metrics.
#    Pick gain/exposure for .env based on brightness / green_cast / sharpness.
make probe

# 5. Run the edge agent (camera -> brain -> keyboard). Keep the console open.
make run
```

Copy `.env.example` to `.env` first and tune camera values from the probe.

## Commands

| Target | What it does |
|---|---|
| `make rebuild` | Stop → rebuild image → start → wait healthy. Always applies full state. |
| `make up` / `make down` | Start / stop stack without rebuilding |
| `make restart-config` | Apply `configs/gestures.yaml` edits without a rebuild |
| `make logs` / `make ps` | Brain logs / stack status |
| `make run` / `make run-bg` | Edge agent in foreground / detached |
| `make test` / `make lint` | Tests / ruff (with security rules) |
| `make probe` | Camera diagnostics + sample shots |
| `make smoke` | Wire-protocol round-trip against the running brain |

## PS3 Eye quality notes

The camera's auto mode in dark scenes pushes gain up: heavy noise and a green
cast. Fixes, in order of impact: **light in front of you** > manual gain/
exposure (`.env`) > built-in frame enhancement (temporal denoise + CLAHE +
sharpening). `make probe` shows raw vs enhanced shots so you can see it.

## Development

- Packages: `src/protocol` (wire contract), `src/brain` (container), `src/edge` (host agent)
- Tests run fully offline on synthetic landmarks: `make test`
- Public repo: read [docs/SECURITY.md](docs/SECURITY.md) before committing —
  secrets never enter this repository, CI enforces it with gitleaks.
