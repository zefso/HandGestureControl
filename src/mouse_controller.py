import pyautogui
import numpy as np

class AirMouse:
    def __init__(self, smoothing=0.3):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = smoothing
        self.is_pressed = False
        
    def move(self, hand_landmarks):
        idx_tip = hand_landmarks.landmark[8]
        curr_x = np.interp(idx_tip.x, [0.25, 0.75], [0, self.screen_w])
        curr_y = np.interp(idx_tip.y, [0.25, 0.65], [0, self.screen_h])
        
        mouse_x = self.prev_x + (curr_x - self.prev_x) * self.smoothing
        mouse_y = self.prev_y + (curr_y - self.prev_y) * self.smoothing
        
        pyautogui.moveTo(mouse_x, mouse_y)
        self.prev_x, self.prev_y = mouse_x, mouse_y

    def handle_actions(self, hand_landmarks):
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        
        dist_idx = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
        dist_mid = ((thumb.x - middle.x)**2 + (thumb.y - middle.y)**2)**0.5
        
        # 1. ЛІВА КНОПКА (Затискання/Драг) - Великий + Вказівний
        if dist_idx < 0.030:
            if not self.is_pressed:
                pyautogui.mouseDown()
                self.is_pressed = True
            return "L_DOWN"
        elif dist_idx > 0.05 and self.is_pressed:
            pyautogui.mouseUp()
            self.is_pressed = False
            return "L_UP"

        # 2. ПРАВА КНОПКА - Великий + Середній
        if dist_mid < 0.030:
            pyautogui.rightClick()
            return "RIGHT_CLICK"

        # 3. СКРОЛІНГ - Якщо вказівний і середній підняті і близько один до одного
        index_up = index.y < hand_landmarks.landmark[6].y
        middle_up = middle.y < hand_landmarks.landmark[10].y
        
        if index_up and middle_up:
            move_y = self.prev_y - index.y * self.screen_h 
            if abs(move_y) > 5:
                scroll_amount = int(move_y / 5)
                pyautogui.scroll(scroll_amount)
                return "SCROLLING"

        return "IDLE"