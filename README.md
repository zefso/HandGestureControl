# Hand Gesture Control System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

[Читати Українською UA](README.uk.md)

> Control your PC with hand gestures in real time — no keyboard, no mouse required.

Built on **MediaPipe** hand tracking + **LSTM** neural network for gesture recognition, with a fully custom HUD and multi-profile action system.

---

## Table of Contents

- [Features](#features)
- [Gestures](#gestures)
- [Modes](#modes)
- [Profiles](#profiles)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Real-time hand tracking** — MediaPipe detects up to 2 hands (21 landmarks each) at 30+ FPS
- **LSTM gesture recognition** — 2-layer LSTM (128 hidden, dropout 0.3) classifies 10 dynamic gestures
- **3 operating modes** — GESTURES, MOUSE, VOLUME, switchable mid-session via gestures
- **AirMouse** — right hand controls cursor; left hand scrolls (position-based) or selects text (fist)
- **Volume control** — left fist + right hand distance adjusts system volume
- **4 action profiles** — `default`, `browser`, `media`, `vscode`; **auto-switches** based on the active window title
- **Configurable via JSON** — change gestures, actions, thresholds, profiles without recompiling
- **Built-in HUD** — live mode/gesture/FPS overlay, settings panel, test mode with per-class probabilities
- **Smart data collector** — variability hints, quality checks, dataset consistency validation
- **Automated tests** — pytest suite covering AirMouse, VolumeController, keypoints, and LSTM model

---

## Gestures

| Gesture | Hand | Description |
|:---|:---:|:---|
| `static` | R | Idle / neutral — no action |
| `swipe_right` | R | Swipe right hand to the right |
| `swipe_left` | L | Swipe left hand to the left |
| `ok` | R | Touch index + thumb → switch to MOUSE mode |
| `stop` | R | Open palm → back to GESTURES mode |
| `browser` | R | Peace sign (index + middle up) |
| `fist_left` | L | Left fist |
| `swipe_up` | R | Swipe right hand upward |
| `swipe_down` | R | Swipe right hand downward |
| `call_me` | R | Pinky + thumb out (phone sign) |

---

## Modes

### GESTURES mode (default)
LSTM recognizes gestures and executes actions from the active profile.
A confirmation buffer (3 of 5 frames) prevents accidental triggers.
Swipes are locked after firing until the hand returns to `static`.

### MOUSE mode
Switch in by holding `ok` gesture for ~7 frames.

| Action | How |
|:---|:---|
| Move cursor | Index finger tip (right hand) |
| Left click | Pinch thumb + index (right hand) |
| Right click | Pinch thumb + middle (right hand) |
| Scroll up/down | Left hand index finger extended — Y position relative to screen center |
| Select text (drag) | Left fist + right hand pinch — cursor moves while LMB is held |

Switch out by holding `stop` gesture.

### VOLUME mode
Only active in GESTURES mode. Hold left fist for ~7 frames, then adjust right-hand thumb-index distance to control system volume.

---

## Profiles

Profiles map gestures to actions. The active profile auto-switches based on the foreground window title every 60 frames.

| Profile | Auto-triggers on |
|:---|:---|
| `default` | any window |
| `browser` | Chrome, Firefox, Edge, Opera, Brave |
| `vscode` | VS Code, PyCharm, IntelliJ |
| `media` | Spotify, VLC, YouTube, Netflix |

Manual override: keys `1`–`4` lock a specific profile (disables auto-switching).

All profiles and their actions are defined in `src/gestures.json` — edit without restarting.

---

## Project Structure

```text
HandGestureControl/
├── src/
│   ├── main.py              # Entry point
│   ├── app.py               # GestureControlApp — main inference loop
│   ├── hud.py               # HUD overlay + settings panel
│   ├── model.py             # GestureLSTM architecture
│   ├── mouse_controller.py  # AirMouse — cursor, clicks, scroll, selection
│   ├── hotkey_executor.py   # GestureActionExecutor — runs profile actions
│   ├── smart_collector.py   # Training data collection tool
│   ├── train.py             # Model training script
│   ├── utils.py             # Keypoint extraction, VolumeController
│   ├── config.py            # Loads gestures.json, exports constants
│   └── gestures.json        # Gestures list, profiles, model + mouse params
├── models/
│   └── gesture_lstm_best.pth  # Trained model weights
├── data/                    # Training sequences (numpy arrays)
├── diagram/                 # Training loss + confusion matrix plots
├── tests/
│   └── test_suite.py        # pytest test suite
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.10+, webcam, Windows (pyautogui + pycaw).

```bash
git clone <repository_url>
cd HandGestureControl

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

---

## Usage

### Run the app

```bash
python src/main.py
```

A window opens showing the camera feed with HUD overlay. Point your right hand at the camera to start.

### Collect training data

```bash
# Collect all gestures (200 samples each by default)
python src/smart_collector.py

# Collect only one gesture
python src/smart_collector.py --target static --count 300

# Clear old dataset and start fresh
python src/smart_collector.py --clean
```

**Controls in collector:** `Space` — record · `N` — next gesture · `P` — prev gesture · `D` — delete last · `Q` — quit

Variability hints cycle automatically — follow them to improve model robustness (distance, angle, speed, lighting).

### Train the model

```bash
python src/train.py
```

- Loads data from `data/`, applies augmentation (noise, scale, time shift)
- Balanced sampling via `WeightedRandomSampler` + `CrossEntropyLoss` with class weights
- Early stopping (patience = 20 epochs by default)
- Best weights saved to `models/gesture_lstm_best.pth`
- Plots saved to `diagram/` (learning curve + confusion matrix)

### Run tests

```bash
pytest tests/
```

---

## Configuration

All runtime parameters live in `src/gestures.json`:

```jsonc
{
  "gestures": ["static", "swipe_right", ...],    // 10 gestures
  "active_profile": "default",
  "profiles": { ... },                            // 4 profiles with per-gesture actions
  "model": {
    "threshold": 0.92,       // confidence cutoff
    "seq_length": 30,        // frames per sequence
    "switch_frames": 7,      // hold frames to switch mode
    "cooldown_frames": 40,   // frames between gesture fires
    "swipe_cooldown_frames": 15
  },
  "mouse": {
    "smoothing": 0.3,        // EMA factor (0 = no movement, 1 = raw)
    "click_zone_x": [0.25, 0.75],
    "click_zone_y": [0.25, 0.65]
  },
  "volume": {
    "smoothing": 0.15,
    "distance_min": 0.02,
    "distance_max": 0.2
  }
}
```

To add a new action to a profile — edit `gestures.json`, no restart needed (call `executor.reload()`).

---

## Keyboard Shortcuts

| Key | Action |
|:---|:---|
| `H` | Toggle help overlay |
| `S` | Open / close settings panel |
| `T` | Test mode — shows per-class probabilities, no actions fired |
| `P` | Pause / Resume |
| `1` – `4` | Switch to profile: default / browser / media / vscode |
| `Q` / `Esc` / `M` | Quit |

---

## Troubleshooting

**Hands not detected**
Improve lighting, avoid strong backlighting, keep hands within the frame.

**Cursor jitter**
Increase `mouse.smoothing` in `gestures.json` (e.g. `0.4`).

**Gestures confused with each other**
Enable test mode (`T`) to see per-class confidence bars, then collect more samples for the confused gesture using `smart_collector.py --target <gesture>`.

**Swipe fires multiple times**
Already handled by `_swipe_locked` flag — swipe re-fires only after the hand returns to `static`. If it still happens, increase `swipe_cooldown_frames` in `gestures.json`.

**`Module not found` on startup**
Activate the virtual environment: `venv\Scripts\activate`, then `pip install -r requirements.txt`.

---

*Diploma project — real-time gesture-based PC control.*
