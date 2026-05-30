"""src/mouse_controller.py — AirMouse: керування курсором і виділенням жестами рук."""
import time
import pyautogui
import numpy as np
from src.config import MOUSE_ZONE_X, MOUSE_ZONE_Y, MOUSE_SMOOTHING


class AirMouse:
    """
    Права рука  — рух курсору (вказівний палець) + ЛКМ/ПКМ
    Ліва рука   — позиційний скролінг (вказівний вгору = скрол у той бік)
    Режим виділення — якщо ліва рука в кулаку під час ЛКМ → курсор рухається
                      при затисненій кнопці (виділення тексту)
    """

    # ── Пороги клавіатури ──────────────────────────────────────────────────────
    _CLICK_FREEZE_DIST = 0.07    # заморозка курсору в зоні передкліку (нормовано)
    _CLICK_DOWN_DIST   = 0.030   # відстань великий–вказівний → mouseDown
    _CLICK_UP_DIST     = 0.055   # відстань при відпусканні  → mouseUp
    _RCLICK_DIST       = 0.032   # великий–середній → правий клік

    # ── Налаштування скролінгу лівою рукою ────────────────────────────────────
    _SCROLL_DEAD    = 0.12   # мертва зона навколо центру екрану (±)
    _SCROLL_MAX_RPS = 25     # макс. подій скролу/с при крайньому положенні
    _SCROLL_CLICKS  = 5      # кількість кліків колеса за одну подію

    def __init__(self,
                 smoothing: float | None = None,
                 zone_x: list | None = None,
                 zone_y: list | None = None):
        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing = smoothing if smoothing is not None else MOUSE_SMOOTHING
        self.zone_x    = zone_x   if zone_x is not None else MOUSE_ZONE_X
        self.zone_y    = zone_y   if zone_y is not None else MOUSE_ZONE_Y

        self.prev_x, self.prev_y = 0.0, 0.0
        self.is_pressed          = False      # зараз натиснута ЛКМ
        self.is_right_pressed    = False      # захист від багаторазового ПКМ
        self._press_is_selection = False      # поточний клік — режим виділення
        self._ls_last_t          = 0.0        # таймер обмеження частоти скролу

    # ── Права рука: рух курсору ───────────────────────────────────────────────

    def move(self, hand_landmarks, selection_mode: bool = False) -> None:
        """Рухає курсор до кінчика вказівного пальця.

        selection_mode=True — курсор рухається навіть коли ЛКМ затиснута
        (потрібно для виділення тексту drag-ом).
        """
        # Заморозка: при звичайному кліку курсор нерухомий під час натискання
        if self.is_pressed and not selection_mode:
            return

        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        dist  = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5

        # Зона передкліку — запобігає стрибку курсора в момент стиснення
        if not self.is_pressed and dist < self._CLICK_FREEZE_DIST:
            return

        curr_x = np.interp(index.x, self.zone_x, [0, self.screen_w])
        curr_y = np.interp(index.y, self.zone_y, [0, self.screen_h])

        # EMA-згладжування для плавного руху
        mx = self.prev_x + (curr_x - self.prev_x) * self.smoothing
        my = self.prev_y + (curr_y - self.prev_y) * self.smoothing
        pyautogui.moveTo(mx, my)
        self.prev_x, self.prev_y = mx, my

    # ── Права рука: кліки ────────────────────────────────────────────────────

    def handle_actions(self, hand_landmarks, selection_mode: bool = False) -> str:
        """Визначає і виконує дії:
          L_DOWN    — ЛКМ натиснута (звичайний клік)
          SELECT    — ЛКМ затиснута у режимі виділення (ліва рука — кулак)
          L_UP      — ЛКМ відпущена
          RIGHT_CLICK — правий клік (великий + середній)
          IDLE      — нічого
        """
        thumb  = hand_landmarks.landmark[4]
        index  = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]

        dist_idx = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        dist_mid = ((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2) ** 0.5

        # ── Обробка поточного натиснення ──────────────────────────────────────
        if self.is_pressed:
            if self._press_is_selection:
                if not selection_mode:
                    # Ліва рука розкрита → завершити виділення
                    pyautogui.mouseUp()
                    self.is_pressed          = False
                    self._press_is_selection = False
                    return 'L_UP'
                return 'SELECT'
            else:
                # Звичайний клік: відпускаємо коли пальці розійшлись
                if dist_idx > self._CLICK_UP_DIST:
                    pyautogui.mouseUp()
                    self.is_pressed = False
                    return 'L_UP'
                return 'L_DOWN'

        # ── Нове натиснення ───────────────────────────────────────────────────
        if dist_idx < self._CLICK_DOWN_DIST:
            pyautogui.mouseDown()
            self.is_pressed          = True
            self._press_is_selection = selection_mode
            return 'SELECT' if selection_mode else 'L_DOWN'

        # ── Правий клік ───────────────────────────────────────────────────────
        if dist_mid < self._RCLICK_DIST:
            if not self.is_right_pressed:
                pyautogui.rightClick()
                self.is_right_pressed = True
            return 'RIGHT_CLICK'
        self.is_right_pressed = False

        return 'IDLE'

    # ── Ліва рука: позиційний скролінг ───────────────────────────────────────

    def handle_left_scroll(self, left_lms) -> tuple[str, float]:
        """Позиційний скролінг лівою рукою.

        Вказівний палець витягнутий:
          - вище центру (0.5) → скрол вгору, швидкість ∝ відстань від центру
          - нижче центру      → скрол вниз
          - мертва зона ±12%  → зупинка

        Повертає (стан, інтенсивність):
          стан        — 'SCROLL_UP' | 'SCROLL_DOWN' | 'IDLE'
          інтенсивність — 0.0 … 1.0 (для відображення смуги швидкості)
        """
        if left_lms is None:
            self._ls_last_t = 0.0
            return 'IDLE', 0.0

        lm    = left_lms.landmark
        index = lm[8]

        # Вказівний палець має бути чітко витягнутий
        if index.y >= lm[6].y:
            self._ls_last_t = 0.0
            return 'IDLE', 0.0

        offset = index.y - 0.5   # від'ємне = вище центру, додатне = нижче

        if abs(offset) < self._SCROLL_DEAD:
            return 'IDLE', 0.0

        # Інтенсивність: 0 на краю мертвої зони → 1 на межі екрану
        intensity = min(1.0, (abs(offset) - self._SCROLL_DEAD)
                        / (0.5 - self._SCROLL_DEAD))

        # Обмеження частоти: чим далі від центру, тим частіше
        interval = 1.0 / max(1, self._SCROLL_MAX_RPS * intensity)
        now      = time.time()
        if now - self._ls_last_t >= interval:
            self._ls_last_t = now
            # від'ємний offset (вище центру) → вгору = позитивний pyautogui.scroll
            amount = self._SCROLL_CLICKS if offset < 0 else -self._SCROLL_CLICKS
            pyautogui.scroll(amount)

        state = 'SCROLL_UP' if offset < 0 else 'SCROLL_DOWN'
        return state, intensity
