# Gesture dictionary

Poses are classified from 21 landmarks; thumb extension is ignored for
fist/palm poses (unreliable at PS3 Eye resolution), but thumb-index proximity
defines the cursor pose.

| Pose | Description |
|---|---|
| `index_up` | only index extended |
| `two_up` | index + middle extended |
| `open_palm` | all four fingers extended |
| `fist` | all four folded |
| `pinch` | thumb tip and index tip close together |

## Default mapping

| Arming pose | Motion | Action | Host effect |
|---|---|---|---|
| `index_up` | swipe right | `alt_tab` | next window |
| `index_up` | swipe left | `alt_shift_tab` | previous window |
| `two_up` | swipe right | `desktop_right` | next virtual desktop |
| `two_up` | swipe left | `desktop_left` | previous virtual desktop |
| `open_palm` | hold ~1s | `pause_system` | toggle pause of gesture control |
| `pinch` | hold (continuous) | cursor steering | mouse follows the thumb-index midpoint |
| `pinch` | quick release-regrip | `left_click` | mouse left click |

## Cursor mode

1. **Enter**: show the pinch gesture (thumb + index tips together).
2. **Steer**: while pinched, the mouse follows the midpoint between thumb and
   index tips (One Euro smoothed, inner 76% of the frame maps to the full
   screen so every pixel is reachable).
3. **Click**: release and re-grip quickly (<380 ms gap) — one left click.
4. **Exit**: relax the pinch for >380 ms, or show an open palm — cursor mode
   ends and swipe gestures re-arm.

While cursor mode is active, swipe gestures are suppressed: the hand that
steers the mouse cannot fire Alt+Tab by accident. The kill switch (open palm
hold) still works.

## Editing gestures

Open the UI (`make run`, then `make ui` → http://127.0.0.1:8756):

- **Gesture map** — every rule as a card; remove with one click
- **Add rule** — pose + motion + action, applied instantly (brain reloads
  `gestures.yaml` over the wire, no rebuild)
- **Custom combos** — *Record key combo* starts a global keyboard hook;
  press any shortcut anywhere (Alt+Tab included), it becomes a reusable
  `custom_N` action; add a rule for it like any other action

Manually: edit `configs/gestures.yaml` (`rules` + `custom_actions`
sections), then `make restart-config` or toggle from the UI.

## Tuning knobs (env, brain)

| Knob | Default | Meaning |
|---|---|---|
| FSM settings | see `neuroint_brain/config.py` | arm/swipe/cooldown/cursor windows |
| `NEUROINT_CAPTURE_FPS` (edge) | 30 | frames sent per second |
| `NEUROINT_JPEG_QUALITY` (edge) | 80 | frame compression |
| `NEUROINT_UI_PORT` (edge) | 8756 | dashboard port (loopback only) |

Roadmap candidates: right click (double regrip), per-app profiles, scroll
wheel gesture, drag-and-drop in cursor mode.

