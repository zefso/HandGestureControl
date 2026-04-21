import pyautogui
import numpy as np


class AirMouse:
    def __init__(self, smoothing: float = 0.3):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = smoothing
        self.is_pressed = False

        self._prev_scroll_y: float | None = None

    def move(self, hand_landmarks) -> None:
        idx_tip = hand_landmarks.landmark[8]
        curr_x = np.interp(idx_tip.x, [0.25, 0.75], [0, self.screen_w])
        curr_y = np.interp(idx_tip.y, [0.25, 0.65], [0, self.screen_h])

        mouse_x = self.prev_x + (curr_x - self.prev_x) * self.smoothing
        mouse_y = self.prev_y + (curr_y - self.prev_y) * self.smoothing

        pyautogui.moveTo(mouse_x, mouse_y)
        self.prev_x, self.prev_y = mouse_x, mouse_y

    def handle_actions(self, hand_landmarks) -> str:
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]

        dist_idx = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        dist_mid = ((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2) ** 0.5

        # 1. LEFT CLICK / DRAG
        if dist_idx < 0.030:
            if not self.is_pressed:
                pyautogui.mouseDown()
                self.is_pressed = True
            self._prev_scroll_y = None  
            return "L_DOWN"

        if dist_idx > 0.05 and self.is_pressed:
            pyautogui.mouseUp()
            self.is_pressed = False
            return "L_UP"

        # 2. RIGHT CLICK
        if dist_mid < 0.030:
            pyautogui.rightClick()
            self._prev_scroll_y = None
            return "RIGHT_CLICK"

        # 3. SCROLLING
        index_up = index.y < hand_landmarks.landmark[6].y
        middle_up = middle.y < hand_landmarks.landmark[10].y
        fingers_close = abs(index.x - middle.x) < 0.05

        if index_up and middle_up and fingers_close:
            if self._prev_scroll_y is not None:
                delta_y = self._prev_scroll_y - index.y  
                scroll_amount = int(delta_y * 30)        
                if abs(scroll_amount) > 0:
                    pyautogui.scroll(scroll_amount)
            self._prev_scroll_y = index.y
            return "SCROLLING"

        self._prev_scroll_y = None
        return "IDLE"