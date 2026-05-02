import cv2
import mediapipe
import pyautogui
import math
import numpy as np
import time
import threading

# ─── Sound feedback (pygame if available, else silent fallback) ───────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)

    def _make_beep(freq, duration_ms, volume=0.4):
        sr = 44100
        n = int(sr * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n, False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767 * volume).astype(np.int16)
        # Fade out to avoid click artifact
        fade = int(n * 0.15)
        wave[-fade:] = (wave[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
        sound = pygame.sndarray.make_sound(wave)
        return sound

    CLICK_SOUND = _make_beep(880, 60)
    DRAG_START_SOUND = _make_beep(660, 80)
    DRAG_END_SOUND = _make_beep(440, 80)
    SOUND_AVAILABLE = True
except Exception:
    SOUND_AVAILABLE = False


def play_sound(sound_name):
    if not SOUND_AVAILABLE:
        return
    sounds = {
        "click": CLICK_SOUND,
        "drag_start": DRAG_START_SOUND,
        "drag_end": DRAG_END_SOUND,
    }
    s = sounds.get(sound_name)
    if s:
        threading.Thread(target=s.play, daemon=True).start()


# ─── Kalman Filter for smooth cursor movement ─────────────────────────────────
class KalmanFilter2D:
    """
    Simple 2D Kalman filter — state: [x, y, vx, vy]
    Smooths raw landmark positions before mapping to screen coords.
    """

    def __init__(self, process_noise=1e-2, measurement_noise=1e1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array(
                [[np.float32(x)], [np.float32(y)], [0.0], [0.0]], dtype=np.float32
            )
            self.initialized = True
        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return float(estimated[0]), float(estimated[1])


# ─── HUD Overlay Renderer ─────────────────────────────────────────────────────
class HUDRenderer:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    # Color palette (BGR)
    C_ACCENT = (0, 255, 180)       # Cyan-green
    C_WARN = (0, 120, 255)         # Orange
    C_DANGER = (50, 50, 255)       # Red
    C_TEXT = (230, 230, 230)       # Off-white
    C_DIM = (100, 100, 100)        # Dim grey
    C_PINCH = (255, 220, 0)        # Gold for pinch

    def __init__(self, win_w, win_h):
        self.w = win_w
        self.h = win_h
        self._gesture_label = "IDLE"
        self._label_color = self.C_DIM
        self._label_ts = 0
        self._label_duration = 0.8  # seconds label stays bright

    def set_gesture(self, label: str, color=None):
        if label != self._gesture_label or time.time() - self._label_ts > self._label_duration:
            self._gesture_label = label
            self._label_color = color or self.C_ACCENT
            self._label_ts = time.time()

    def _alpha_rect(self, img, x1, y1, x2, y2, color, alpha=0.45):
        sub = img[y1:y2, x1:x2]
        rect = np.full(sub.shape, color, dtype=np.uint8)
        cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
        img[y1:y2, x1:x2] = sub

    def draw_corner_brackets(self, img, x, y, w, h, color, thickness=2, length=16):
        pts = [
            ((x, y), (x + length, y), (x, y + length)),
            ((x + w, y), (x + w - length, y), (x + w, y + length)),
            ((x, y + h), (x + length, y + h), (x, y + h - length)),
            ((x + w, y + h), (x + w - length, y + h), (x + w, y + h - length)),
        ]
        for corner, p1, p2 in pts:
            cv2.line(img, corner, p1, color, thickness)
            cv2.line(img, corner, p2, color, thickness)

    def draw_gesture_badge(self, img):
        label = self._gesture_label
        age = time.time() - self._label_ts
        if age > self._label_duration:
            color = self.C_DIM
        else:
            fade = max(0, 1 - age / self._label_duration)
            base = np.array(self._label_color, dtype=np.float32)
            dim = np.array(self.C_DIM, dtype=np.float32)
            color = tuple((base * fade + dim * (1 - fade)).astype(int).tolist())

        # Background pill
        (tw, _), _ = cv2.getTextSize(label, self.FONT_BOLD, 0.9, 2)
        px, py = 16, 16
        bx1, by1 = px, py
        bx2, by2 = px + tw + 28, py + 38
        self._alpha_rect(img, bx1, by1, bx2, by2, (20, 20, 20), alpha=0.6)
        self.draw_corner_brackets(img, bx1, by1, bx2 - bx1, by2 - by1, color, thickness=2)
        cv2.putText(img, label, (px + 12, py + 26), self.FONT_BOLD, 0.9, color, 2, cv2.LINE_AA)

    def draw_dist_bar(self, img, dist, threshold_click=23, threshold_drag=40):
        bw, bh = 160, 8
        bx, by = self.w - bw - 16, self.h - 40

        # Label
        cv2.putText(img, "PINCH DIST", (bx, by - 8), self.FONT, 0.38, self.C_DIM, 1, cv2.LINE_AA)

        # Track
        self._alpha_rect(img, bx, by, bx + bw, by + bh, (40, 40, 40), alpha=0.7)

        # Fill
        fill = min(int(dist / 80 * bw), bw)
        if dist < threshold_click:
            bar_color = self.C_DANGER
        elif dist < threshold_drag:
            bar_color = self.C_WARN
        else:
            bar_color = self.C_ACCENT
        if fill > 0:
            cv2.rectangle(img, (bx, by), (bx + fill, by + bh), bar_color, -1)

        # Threshold ticks
        t1x = bx + int(threshold_click / 80 * bw)
        t2x = bx + int(threshold_drag / 80 * bw)
        cv2.line(img, (t1x, by - 3), (t1x, by + bh + 3), self.C_DANGER, 1)
        cv2.line(img, (t2x, by - 3), (t2x, by + bh + 3), self.C_WARN, 1)

        # Value
        cv2.putText(img, f"{dist:.0f}px", (bx + bw + 6, by + bh),
                    self.FONT, 0.38, self.C_TEXT, 1, cv2.LINE_AA)

    def draw_fps(self, img, fps):
        cv2.putText(img, f"FPS {fps:.0f}", (self.w - 70, 28),
                    self.FONT, 0.5, self.C_DIM, 1, cv2.LINE_AA)

    def draw_no_hand(self, img):
        msg = "NO HAND DETECTED"
        tw = cv2.getTextSize(msg, self.FONT_BOLD, 0.7, 1)[0][0]
        cv2.putText(img, msg, ((self.w - tw) // 2, self.h - 20),
                    self.FONT_BOLD, 0.7, self.C_DIM, 1, cv2.LINE_AA)

    def draw_drag_indicator(self, img, dragging):
        if dragging:
            label = "● DRAG ACTIVE"
            self._alpha_rect(img, 16, self.h - 50, 180, self.h - 24, (20, 20, 20), 0.6)
            cv2.putText(img, label, (22, self.h - 32),
                        self.FONT_BOLD, 0.55, self.C_WARN, 1, cv2.LINE_AA)

    def draw_landmark_highlight(self, img, x, y, landmark_id, color):
        """Draw a highlighted ring around a specific landmark."""
        radius = 12 if landmark_id in (8, 4) else 6
        cv2.circle(img, (x, y), radius + 4, (*color[:3],), 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)

    def draw_pinch_line(self, img, p1, p2, dist, threshold):
        color = self.C_DANGER if dist < threshold else self.C_PINCH
        cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.circle(img, mid, 4, color, -1)


# ─── Main ─────────────────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mediapipe.solutions.hands
mp_draw = mediapipe.solutions.drawing_utils
mp_style = mediapipe.solutions.drawing_styles

capture_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6,
)

screen_width, screen_height = pyautogui.size()

kalman = KalmanFilter2D(process_noise=5e-3, measurement_noise=5.0)

camera = cv2.VideoCapture(0)
cam_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

hud = HUDRenderer(cam_w, cam_h)

dragging = False
x1 = x2 = y1 = y2 = 0

# FPS tracking
fps_history = []
prev_time = time.time()

# Cooldown to prevent click-spam
last_click_time = 0
CLICK_COOLDOWN = 0.35  # seconds

print("Gesture Mouse started. Press X to quit.")

while True:
    ret, image = camera.read()
    if not ret:
        break

    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps_history.append(1.0 / max(dt, 1e-6))
    if len(fps_history) > 20:
        fps_history.pop(0)
    fps = sum(fps_history) / len(fps_history)

    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks

    hand_detected = False
    dist = 999

    if all_hands:
        hand_detected = True
        for hand in all_hands:
            # Draw styled skeleton
            mp_draw.draw_landmarks(
                image,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(80, 80, 80), thickness=1),
            )

            one_hand_landmarks = hand.landmark
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                if id == 8:  # Index fingertip
                    # Apply Kalman filter
                    kx, ky = kalman.update(x, y)
                    mouse_x = int(screen_width / image_width * kx)
                    mouse_y = int(screen_height / image_height * ky)
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1, y1 = x, y
                    hud.draw_landmark_highlight(image, x, y, 8, (0, 255, 180))

                if id == 4:  # Thumb tip
                    x2, y2 = x, y
                    hud.draw_landmark_highlight(image, x, y, 4, (255, 220, 0))

            # Pinch distance
            dist = math.hypot(x2 - x1, y2 - y1)

            # Draw pinch line between thumb and index
            hud.draw_pinch_line(image, (x1, y1), (x2, y2), dist, threshold=40)

            # ── Gesture logic ──────────────────────────────────────────
            if dist < 23:
                if now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time = now
                    play_sound("click")
                hud.set_gesture("CLICK", hud.C_DANGER)

            elif dist < 40:
                if not dragging:
                    dragging = True
                    pyautogui.mouseDown()
                    play_sound("drag_start")
                hud.set_gesture("DRAG", hud.C_WARN)

            else:
                if dragging:
                    dragging = False
                    pyautogui.mouseUp()
                    play_sound("drag_end")
                hud.set_gesture("MOVE", hud.C_ACCENT)

    else:
        if dragging:
            dragging = False
            pyautogui.mouseUp()
        hud.set_gesture("IDLE", hud.C_DIM)

    # ── HUD rendering ──────────────────────────────────────────────────
    hud.draw_gesture_badge(image)
    hud.draw_fps(image, fps)
    hud.draw_dist_bar(image, dist if hand_detected else 999)
    hud.draw_drag_indicator(image, dragging)
    if not hand_detected:
        hud.draw_no_hand(image)

    cv2.imshow("GestureOS - Hand Tracking", image)
    key = cv2.waitKey(1)
    if key in (ord("x"), ord("X")):
        break

if dragging:
    pyautogui.mouseUp()
camera.release()
cv2.destroyAllWindows()
print("GestureOS closed.")
