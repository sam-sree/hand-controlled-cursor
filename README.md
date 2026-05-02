# GestureOS — Hand Tracking Mouse Controller

> Control your computer with nothing but your hand. No mouse. No touch screen. Just a webcam.

GestureOS is a real-time hand gesture system that replaces your mouse using computer vision. Built with MediaPipe, OpenCV, and PyAutoGUI, it tracks your fingers frame-by-frame and maps natural hand movements to cursor control, clicking, and dragging — with a live heads-up display and audio feedback.

---

## Demo

> Point your index finger to move. Pinch to click. Hold the pinch to drag.

```
[ Your webcam feed with hand skeleton overlay ]
[ MOVE / CLICK / DRAG badge — top left        ]
[ PINCH DIST bar — bottom right               ]
[ FPS counter — top right                     ]
```

---

## Features

- **Real-time hand tracking** via MediaPipe (21 landmarks, up to 30+ FPS)
- **Kalman filter cursor smoothing** — eliminates jitter using a 4-state `[x, y, vx, vy]` model; no more shaky cursor from minor finger tremors
- **Three gesture modes** — MOVE, CLICK, DRAG — detected from pinch distance between index fingertip and thumb tip
- **Click cooldown** — 350ms debounce prevents accidental double-clicks during pinch hold
- **Live HUD overlay** drawn directly on the webcam feed:
  - Gesture badge (top-left) with fade animation
  - Pinch distance bar with color-coded thresholds (green → orange → red)
  - Threshold tick marks showing exact click and drag boundaries
  - FPS counter
  - Drag active indicator
  - "NO HAND DETECTED" warning when hand leaves frame
- **Sound feedback** synthesized procedurally via pygame — no audio files needed:
  - 880 Hz click tone (60ms)
  - 660 Hz drag-start tone (80ms)
  - 440 Hz drag-end tone (80ms)
- **Silent fallback** if pygame is not installed — everything else works normally
- Plays sounds on a daemon thread — audio never blocks the video loop

---

## How It Works

```
Webcam frame
    │
    ▼
MediaPipe Hands  ──►  21 hand landmarks (x, y per landmark)
    │
    ▼
Extract landmark 8 (index tip) + landmark 4 (thumb tip)
    │
    ▼
Kalman Filter  ──►  smoothed (x, y) position
    │
    ▼
Map to screen coordinates  ──►  PyAutoGUI moveTo()
    │
    ▼
Compute pinch distance = hypot(index_tip - thumb_tip)
    │
    ├── dist < 23px   ──►  CLICK  (with 350ms cooldown)
    ├── dist < 40px   ──►  DRAG   (mouseDown / mouseUp on transition)
    └── dist >= 40px  ──►  MOVE
    │
    ▼
HUD overlay rendered on frame  ──►  cv2.imshow()
```

### Kalman Filter

The filter maintains a state vector `[x, y, vx, vy]` and uses constant-velocity motion as its prediction model. On each frame, it predicts where the finger will be, then corrects against the raw MediaPipe measurement. The result is a smooth trajectory that reacts quickly to intentional movement but ignores high-frequency noise.

Tunable parameters at the top of `gesture_mouse.py`:

| Parameter | Default | Effect |
|---|---|---|
| `process_noise` | `5e-3` | Higher = follows raw input more closely |
| `measurement_noise` | `5.0` | Higher = smoother but laggier |

---

## Gesture Reference

| Gesture | How to perform | What happens |
|---|---|---|
| **MOVE** | Extend index finger, keep thumb away | Cursor follows index fingertip |
| **CLICK** | Pinch index + thumb (distance < 23px) | Single left click |
| **DRAG** | Partial pinch (distance 23–40px), then move | mouseDown held while moving |
| **Release drag** | Open hand (distance ≥ 40px) | mouseUp |

> **Tip:** Practice the click gesture in the air before trying it on screen. Watch the PINCH DIST bar to learn exactly how far to pinch.

---

## Requirements

- Python 3.8+
- Webcam (built-in or USB)
- Windows / macOS / Linux

### Dependencies

```
opencv-python
mediapipe
pyautogui
numpy
pygame          # optional — for sound feedback
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/gestureos.git
cd gestureos

# 2. Install dependencies
pip install opencv-python mediapipe pyautogui numpy

# 3. (Optional) Install pygame for sound feedback
pip install pygame

# 4. Run
python gesture_mouse.py
```

Press **X** to quit.

---

## HUD Reference

| Element | Location | Description |
|---|---|---|
| Gesture badge | Top-left | Current gesture state with fade glow |
| FPS counter | Top-right | Rolling average over last 20 frames |
| Pinch distance bar | Bottom-right | Live bar — green=move, orange=drag, red=click |
| Drag indicator | Bottom-left | Appears only while drag is active |
| Hand skeleton | Overlay | Styled landmark + connection drawing |
| Cyan ring | Index fingertip | Landmark 8 — controls cursor |
| Gold ring | Thumb tip | Landmark 4 — pinch reference point |
| Pinch line | Between fingertips | Gold normally, red when pinching |

---

## Troubleshooting

**Cursor is jittery**
Increase `measurement_noise` in `KalmanFilter2D(process_noise=5e-3, measurement_noise=5.0)`. Try values between `5.0` and `30.0`.

**Cursor lags behind my finger**
Decrease `measurement_noise` or increase `process_noise`. Try `measurement_noise=1.0`.

**Clicks are firing too fast**
Increase `CLICK_COOLDOWN` from `0.35` to `0.5` or higher.

**Hand not detected**
- Ensure adequate, even lighting on your hand
- Keep hand within the camera frame and avoid cluttered backgrounds
- Check that your webcam index is correct — change `cv2.VideoCapture(0)` to `VideoCapture(1)` if using an external webcam

**Wrong monitor / cursor jumps to corner**
PyAutoGUI maps the full camera frame to your primary monitor. For multi-monitor setups, adjust the coordinate mapping in the `id == 8` block.

**No sound**
Run `pip install pygame`. If already installed, check that your system audio is not muted. The program will still work without sound.

**MediaPipe warnings on startup**
The `inference_feedback_manager` warnings are benign — they come from MediaPipe internals and do not affect functionality.

---

## Project Structure

```
gestureos/
├── gesture_mouse.py    # Main script — all logic self-contained
└── README.md
```

---

## Technologies

| Library | Role |
|---|---|
| [MediaPipe](https://google.github.io/mediapipe/) | Hand landmark detection (21 points) |
| [OpenCV](https://opencv.org/) | Webcam capture, frame rendering, Kalman filter |
| [PyAutoGUI](https://pyautogui.readthedocs.io/) | Cross-platform mouse control |
| [NumPy](https://numpy.org/) | Signal processing, sound synthesis |
| [pygame](https://www.pygame.org/) | Audio playback (optional) |

---

## Roadmap

- [ ] Scroll gesture (two-finger vertical swipe)
- [ ] Right-click gesture (ring finger tap)
- [ ] Palm-open pause mode (prevent false inputs)
- [ ] Dwell clicking for accessibility (hover to click without pinch)
- [ ] Config file (YAML) for sensitivity and threshold tuning
- [ ] Multi-monitor coordinate mapping
- [ ] Two-hand support

---

