import pyautogui
import numpy as np
from src.config import MOUSE_ZONE_X, MOUSE_ZONE_Y, MOUSE_SMOOTHING


class AirMouse:
    """
    Керування мишею через положення правої руки.

    Зони захоплення (zone_x, zone_y) визначають яка частина кадру
    камери маппується на весь екран. Береться з config.py за замовчуванням.
    """

    def __init__(self,
                 smoothing: float | None = None,
                 zone_x: list | None = None,
                 zone_y: list | None = None):
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing = smoothing if smoothing is not None else MOUSE_SMOOTHING
        self.zone_x    = zone_x    if zone_x    is not None else MOUSE_ZONE_X
        self.zone_y    = zone_y    if zone_y    is not None else MOUSE_ZONE_Y

        self.prev_x, self.prev_y = 0, 0
        self.is_pressed = False
        self._prev_scroll_y: float | None = None

    # Якщо відстань великий-вказівний менша за цей поріг —
    # курсор заморожується, щоб клік не зміщував позицію.
    _CLICK_FREEZE_DIST = 0.07

    def move(self, hand_landmarks) -> None:
        """Переміщує курсор до позиції вказівного пальця (landmark 8).
        Курсор не рухається під час натискання та в зоні підходу до кліку,
        щоб уникнути стрибка курсора вниз при стисканні пальців.
        """
        # Заморожуємо курсор поки кнопка натиснута
        if self.is_pressed:
            return

        # Заморожуємо коли пальці вже наближаються (pre-click зона)
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        dist  = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        if dist < self._CLICK_FREEZE_DIST:
            return

        idx_tip = hand_landmarks.landmark[8]
        curr_x = np.interp(idx_tip.x, self.zone_x, [0, self.screen_w])
        curr_y = np.interp(idx_tip.y, self.zone_y, [0, self.screen_h])

        # EMA-згладжування для плавного руху
        mouse_x = self.prev_x + (curr_x - self.prev_x) * self.smoothing
        mouse_y = self.prev_y + (curr_y - self.prev_y) * self.smoothing

        pyautogui.moveTo(mouse_x, mouse_y)
        self.prev_x, self.prev_y = mouse_x, mouse_y

    def handle_actions(self, hand_landmarks) -> str:
        """
        Визначає і виконує дії миші за жестами:
          L_DOWN       — великий + вказівний ближче 0.030 (затиснути)
          L_UP         — великий + вказівний далі 0.050  (відпустити)
          RIGHT_CLICK  — великий + середній ближче 0.030
          SCROLLING    — два підняті пальці поряд, рух вгору/вниз
          IDLE         — інакше
        """
        thumb  = hand_landmarks.landmark[4]
        index  = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]

        dist_idx = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        dist_mid = ((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2) ** 0.5

        # 1. ЛІВИЙ КЛІК / DRAG
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

        # 2. ПРАВИЙ КЛІК
        if dist_mid < 0.030:
            pyautogui.rightClick()
            self._prev_scroll_y = None
            return "RIGHT_CLICK"

        # 3. СКРОЛІНГ — вказівний і середній підняті і близько
        index_up     = index.y  < hand_landmarks.landmark[6].y
        middle_up    = middle.y < hand_landmarks.landmark[10].y
        fingers_close = abs(index.x - middle.x) < 0.05

        if index_up and middle_up and fingers_close:
            if self._prev_scroll_y is not None:
                delta_y      = self._prev_scroll_y - index.y
                scroll_amount = int(delta_y * 30)
                if abs(scroll_amount) > 0:
                    pyautogui.scroll(scroll_amount)
            self._prev_scroll_y = index.y
            return "SCROLLING"

        self._prev_scroll_y = None
        return "IDLE"