import time
import pyautogui
import numpy as np
from src.config import MOUSE_ZONE_X, MOUSE_ZONE_Y, MOUSE_SMOOTHING


class AirMouse:
    """
    Right hand — cursor (index fingertip) + click/right-click
    Left hand  — position-based scroll: index finger above center = scroll up,
                 below center = scroll down; speed proportional to distance.
    """

    _CLICK_FREEZE_DIST = 0.07    # pre-click cursor freeze (normalised)
    _CLICK_DOWN_DIST   = 0.030   # index–thumb pinch → mouseDown
    _CLICK_UP_DIST     = 0.055   # release threshold → mouseUp
    _RCLICK_DIST       = 0.032   # middle–thumb pinch → rightClick

    # Left-hand scroll tuning
    _SCROLL_DEAD       = 0.12    # dead-zone half-width around screen centre
    _SCROLL_MAX_RPS    = 18      # max scroll events per second (at finger extreme)
    _SCROLL_CLICKS     = 3       # wheel clicks per scroll event

    def __init__(self,
                 smoothing: float | None = None,
                 zone_x: list | None = None,
                 zone_y: list | None = None):
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing = smoothing if smoothing is not None else MOUSE_SMOOTHING
        self.zone_x    = zone_x   if zone_x is not None else MOUSE_ZONE_X
        self.zone_y    = zone_y   if zone_y is not None else MOUSE_ZONE_Y

        self.prev_x, self.prev_y = 0.0, 0.0
        self.is_pressed          = False
        self.is_right_pressed    = False
        self._ls_last_t          = 0.0   # left-scroll rate limiter

    # ── Right hand: cursor ────────────────────────────────────────────────────

    def move(self, hand_landmarks) -> None:
        """Move cursor to index fingertip.  Freezes during press and in pre-click zone."""
        if self.is_pressed:
            return

        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        dist  = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        if dist < self._CLICK_FREEZE_DIST:
            return

        curr_x = np.interp(index.x, self.zone_x, [0, self.screen_w])
        curr_y = np.interp(index.y, self.zone_y, [0, self.screen_h])
        mx = self.prev_x + (curr_x - self.prev_x) * self.smoothing
        my = self.prev_y + (curr_y - self.prev_y) * self.smoothing
        pyautogui.moveTo(mx, my)
        self.prev_x, self.prev_y = mx, my

    # ── Right hand: click / right-click ───────────────────────────────────────

    def handle_actions(self, hand_landmarks) -> str:
        """Returns L_DOWN | L_UP | RIGHT_CLICK | IDLE."""
        thumb  = hand_landmarks.landmark[4]
        index  = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]

        dist_idx = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        dist_mid = ((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2) ** 0.5

        # Left click
        if dist_idx < self._CLICK_DOWN_DIST:
            if not self.is_pressed:
                pyautogui.mouseDown()
                self.is_pressed = True
            return 'L_DOWN'

        if dist_idx > self._CLICK_UP_DIST and self.is_pressed:
            pyautogui.mouseUp()
            self.is_pressed = False
            return 'L_UP'

        # Right click
        if dist_mid < self._RCLICK_DIST:
            if not self.is_right_pressed:
                pyautogui.rightClick()
                self.is_right_pressed = True
            return 'RIGHT_CLICK'
        self.is_right_pressed = False

        return 'IDLE'

    # ── Left hand: position-based scroll ──────────────────────────────────────

    def handle_left_scroll(self, left_lms) -> tuple[str, float]:
        """
        Position-based scroll using left index finger.

        Returns (state, intensity):
          state     — 'SCROLL_UP' | 'SCROLL_DOWN' | 'IDLE'
          intensity — 0.0 … 1.0 (how far from center, for HUD bar)
        """
        if left_lms is None:
            self._ls_last_t = 0.0
            return 'IDLE', 0.0

        lm    = left_lms.landmark
        index = lm[8]

        # Index finger must be clearly extended
        if index.y >= lm[6].y:
            self._ls_last_t = 0.0
            return 'IDLE', 0.0

        offset = index.y - 0.5        # <0 above center, >0 below center
        if abs(offset) < self._SCROLL_DEAD:
            return 'IDLE', 0.0

        # Intensity: 0 at dead-zone edge → 1 at screen top/bottom
        intensity = min(1.0, (abs(offset) - self._SCROLL_DEAD)
                        / (0.5 - self._SCROLL_DEAD))

        # Rate limit: faster when finger is further from centre
        interval = 1.0 / max(1, self._SCROLL_MAX_RPS * intensity)
        now = time.time()
        if now - self._ls_last_t >= interval:
            self._ls_last_t = now
            # negative offset (above centre) → scroll up (+)
            amount = self._SCROLL_CLICKS if offset < 0 else -self._SCROLL_CLICKS
            pyautogui.scroll(amount)

        state = 'SCROLL_UP' if offset < 0 else 'SCROLL_DOWN'
        return state, intensity
