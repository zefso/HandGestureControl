"""src/hud.py — HUD, settings panel, and draw helpers."""
import cv2
import numpy as np
import time

from src.config import (
    GESTURES, THRESHOLD, COOLDOWN_FRAMES, SWITCH_FRAMES,
)

# ── Color palette ──────────────────────────────────────────────────────────────
C: dict[str, tuple] = {
    'bg':      (12,  12,  20),
    'bg2':     (22,  22,  38),
    'bg3':     (38,  38,  62),
    'border':  (58,  58,  95),
    'accent':  (0,   200, 255),
    'green':   (30,  210,  70),
    'orange':  (0,   160, 255),
    'purple':  (220,  60, 220),
    'red':     (60,   60, 230),
    'white':   (240, 240, 245),
    'gray':    (130, 130, 160),
    'dark':    (42,  42,  68),
    'yellow':  (0,   230, 230),
}

MODE_COLORS: dict[str, tuple] = {
    'GESTURES': (0,   200, 255),
    'MOUSE':    (220,  60, 220),
    'VOLUME':   (30,  210,  70),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Gesture guide shown in the settings panel
GESTURE_GUIDE = [
    ('static',      'R', 'Idle / neutral'),
    ('ok',          'R', 'Switch to MOUSE mode'),
    ('stop',        'R', 'Back to GESTURES mode'),
    ('browser',     'R', 'V-sign: index + middle up'),
    ('swipe_right', 'R', 'Swipe right hand rightward'),
    ('swipe_left',  'L', 'Swipe left hand leftward'),
    ('fist_left',   'L', 'Make a fist with left hand'),
    ('thumbs_up',   'R', 'Thumb pointing upward'),
]

SHORTCUTS = [
    ('H',       'Help overlay on / off'),
    ('T',       'Test mode (no actions fired)'),
    ('P',       'Pause / Resume'),
    ('M',       'Back to menu'),
    ('Q / ESC', 'Quit'),
    ('1-4',     'Manual profile override'),
    ('S',       'Open / close this panel'),
]


# ── Draw helpers ───────────────────────────────────────────────────────────────

def _rr_fill(img, x1, y1, x2, y2, color, r: int):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


def rr(img, x1, y1, x2, y2, color, r: int = 10, alpha: float = 1.0):
    """Rounded rectangle with optional transparency."""
    if alpha < 1.0:
        ov = img.copy()
        _rr_fill(ov, x1, y1, x2, y2, color, r)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    else:
        _rr_fill(img, x1, y1, x2, y2, color, r)


def txt(img, text: str, pos: tuple, scale: float = 0.52,
        color=None, bold: bool = False, anchor: str = 'left'):
    color  = color or C['white']
    weight = 2 if bold else 1
    if anchor != 'left':
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, weight)
        if anchor == 'center':
            pos = (pos[0] - tw // 2, pos[1] + th // 2)
        else:
            pos = (pos[0] - tw, pos[1])
    cv2.putText(img, text, pos, _FONT, scale, color, weight, cv2.LINE_AA)


def tsize(text: str, scale: float = 0.52, bold: bool = False) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, _FONT, scale, 2 if bold else 1)
    return w, h


def _semi(img, x1: int, y1: int, x2: int, y2: int, color, alpha: float = 0.85):
    """Semi-transparent rectangle (addWeighted blend)."""
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def draw_gear(img, cx: int, cy: int,
              r_out: int = 13, r_in: int = 8, n: int = 6,
              color=(200, 200, 210)):
    """Gear icon drawn as a polygon + hollow centre."""
    n4     = n * 4
    angles = [i * 2 * np.pi / n4 - np.pi / 2 for i in range(n4)]
    pts    = [[int(cx + (r_out if i % 4 < 2 else r_in) * np.cos(a)),
               int(cy + (r_out if i % 4 < 2 else r_in) * np.sin(a))]
              for i, a in enumerate(angles)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)
    cv2.circle(img, (cx, cy), max(1, r_in - 3), C['bg2'], -1)
    cv2.circle(img, (cx, cy), max(1, r_in - 3), color, 1)


# ── HUD ───────────────────────────────────────────────────────────────────────

class HUD:
    """
    Renders all on-screen elements on top of the camera frame.

    Top bar    — mode badge, gesture name, profile indicator, FPS, gear button
    Bottom bar — last action, gesture history chips
    Right strip — confidence + cooldown bars (vertical)
    Settings panel — slides in from right when gear/S pressed
    Help, flash, volume, test panel, switch progress overlays
    """

    TOP_H   = 52
    BOT_H   = 56
    GEAR_SZ = 38

    SETTINGS_PW = 320

    def __init__(self):
        self.help_on       = False
        self.settings_open = False
        self._flash_msg    = ''
        self._flash_end    = 0.0
        self._gear_rect    = (0, 0, 1, 1)
        self._x_rect       = (0, 0, 1, 1)   # red X button rect
        self._s_btns: dict = {}              # settings internal buttons
        self._scroll       = 0              # scroll offset (px)
        self._scroll_max   = 0              # set after each draw
        self._cameras: list        = []
        self._resolutions: list    = []
        # Pending changes read by the app each frame
        self.selected_cam: int | None   = None
        self.selected_res: tuple | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def flash(self, msg: str, sec: float = 1.6):
        self._flash_msg = msg
        self._flash_end = time.time() + sec

    def toggle_help(self):
        self.help_on = not self.help_on

    def toggle_settings(self):
        self.settings_open = not self.settings_open
        if self.settings_open:
            self.help_on = False

    def is_gear_click(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self._gear_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def handle_click(self, x: int, y: int) -> None:
        """Route a left-button click to gear, X button, or settings controls."""
        if self.is_gear_click(x, y):
            self.toggle_settings()
            return
        if not self.settings_open:
            return
        # Red X button
        x1, y1, x2, y2 = self._x_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.settings_open = False
            return
        # Settings internal buttons (cam / res)
        for name, (bx1, by1, bx2, by2) in self._s_btns.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self._on_settings_click(name)
                return

    def _on_settings_click(self, name: str):
        if name.startswith('cam_'):
            self.selected_cam = int(name[4:])
        elif name.startswith('res_') and self._resolutions:
            i = int(name[4:])
            self.selected_res = self._resolutions[i][0]

    def handle_scroll(self, delta: int):
        """Scroll the settings panel content. delta > 0 = scroll down."""
        if self.settings_open:
            self._scroll = max(0, min(self._scroll + delta, self._scroll_max))

    # ── Main draw ─────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, *,
             mode: str, profile: str, gesture: str, confidence: float,
             last_action: str, history, fps: float,
             cooldown: int, switch_cnt: int, switching: bool,
             paused: bool = False, vol: int | None = None,
             test_mode: bool = False, all_probs: np.ndarray | None = None,
             auto_profile: bool = True,
             cameras: list | None = None, cur_cam: int = 0,
             resolutions: list | None = None, cur_res: tuple = (1280, 720)):

        # Cache for use by settings panel
        self._cameras    = cameras or []
        self._resolutions = resolutions or []

        h, w = frame.shape[:2]
        mc   = MODE_COLORS.get(mode, C['accent'])

        self._draw_topbar(frame, w, h, mode, mc, profile, gesture,
                          confidence, fps, test_mode, auto_profile)
        self._draw_bottombar(frame, w, h, last_action, history)
        self._draw_conf_strip(frame, w, h, confidence, cooldown, mode)

        if paused:
            self._draw_paused(frame, w, h)
        if switching:
            self._draw_switch_progress(frame, w, h, mode, mc, switch_cnt)
        if vol is not None:
            self._draw_volume(frame, h, vol)
        if test_mode and all_probs is not None:
            self._draw_test_panel(frame, w, all_probs)
        if time.time() < self._flash_end:
            self._draw_flash(frame, w, h, mc)
        if self.settings_open:
            self._draw_settings(frame, w, h, cur_cam, cur_res)
        elif self.help_on:
            self._draw_help(frame, w, h)

    # ── Top bar ───────────────────────────────────────────────────────────────

    def _draw_topbar(self, f, w, h, mode, mc, profile, gesture,
                     confidence, fps, test_mode, auto_profile):
        TH = self.TOP_H
        _semi(f, 0, 0, w, TH, C['bg'], alpha=0.88)
        cv2.line(f, (0, TH), (w, TH), mc, 2)

        # Mode badge (left)
        badge   = f' {mode} '
        bw, bh  = tsize(badge, 0.6, bold=True)
        pad     = 8
        rr(f, 10, pad, 10 + bw + 18, TH - pad, mc, r=7)
        txt(f, badge, (19, pad + bh + 2), scale=0.6, color=C['bg'], bold=True)

        x_cur = 10 + bw + 24

        # TEST badge
        if test_mode:
            tw, th = tsize(' TEST ', 0.46, bold=True)
            rr(f, x_cur, pad + 2, x_cur + tw + 12, TH - pad - 2, C['orange'], r=6)
            txt(f, ' TEST ', (x_cur + 6, pad + th + 4),
                scale=0.46, color=C['bg'], bold=True)
            x_cur += tw + 18

        # Active gesture name
        if gesture not in ('static', '') and confidence > 0.5:
            gname = gesture.replace('_', ' ').upper()
            txt(f, gname, (x_cur, TH // 2 + 9),
                scale=0.62, color=mc, bold=True)

        # Profile indicator (center)
        tag_col   = C['gray'] if auto_profile else C['yellow']
        tag_text  = 'AUTO' if auto_profile else 'MANUAL'
        ptw, _    = tsize(profile.upper(), 0.44, bold=True)
        cx        = w // 2
        txt(f, profile.upper(), (cx, TH // 2 + 7),
            scale=0.44, color=C['white'], bold=True, anchor='center')
        txt(f, tag_text, (cx + ptw // 2 + 10, TH // 2 + 7),
            scale=0.31, color=tag_col)

        # FPS
        gs      = self.GEAR_SZ
        fps_col = C['green'] if fps >= 25 else C['orange'] if fps >= 15 else C['red']
        gear_x1 = w - gs - 10
        ftw, _  = tsize(f'{fps:.0f} FPS', 0.43)
        txt(f, f'{fps:.0f} FPS', (gear_x1 - ftw - 12, TH // 2 + 7),
            scale=0.43, color=fps_col)

        # Gear button
        gear_y1 = (TH - gs) // 2
        gear_x2 = w - 10
        gear_y2 = gear_y1 + gs
        gcol    = C['accent'] if self.settings_open else C['bg3']
        rr(f, gear_x1, gear_y1, gear_x2, gear_y2, gcol, r=8)
        draw_gear(f, (gear_x1 + gear_x2) // 2, (gear_y1 + gear_y2) // 2,
                  color=C['bg'] if self.settings_open else C['accent'])
        self._gear_rect = (gear_x1, gear_y1, gear_x2, gear_y2)

    # ── Bottom bar ────────────────────────────────────────────────────────────

    def _draw_bottombar(self, f, w, h, last_action, history):
        BH = self.BOT_H
        _semi(f, 0, h - BH, w, h, C['bg'], alpha=0.88)
        cv2.line(f, (0, h - BH), (w, h - BH), C['border'], 1)

        # Arrow + last action
        txt(f, '>', (14, h - BH // 2 + 7), scale=0.52, color=C['accent'], bold=True)
        txt(f, last_action[:58], (34, h - BH // 2 + 7), scale=0.48, color=C['white'])

        # History chips (right-to-left)
        chips = list(history)[-5:]
        cx    = w - 18
        for g in reversed(chips):
            tw, _  = tsize(g, 0.33)
            chip_w = tw + 14
            cx    -= chip_w
            rr(f, cx, h - BH + 10, cx + chip_w, h - 12, C['bg3'], r=5)
            txt(f, g, (cx + 7, h - 14), scale=0.33, color=C['gray'])
            cx -= 7

    # ── Right confidence strip ────────────────────────────────────────────────

    def _draw_conf_strip(self, f, w, h, confidence, cooldown, mode):
        if mode != 'GESTURES':
            return
        TH, BH    = self.TOP_H, self.BOT_H
        strip_h   = h - TH - BH
        sx        = w - 9
        sy        = TH

        cv2.rectangle(f, (sx, sy), (w, sy + strip_h), C['dark'], -1)

        col  = (C['green']  if confidence >= THRESHOLD else
                C['orange'] if confidence >= 0.6       else C['red'])
        fill = int(confidence * strip_h)
        if fill > 0:
            cv2.rectangle(f, (sx, sy + strip_h - fill), (w, sy + strip_h), col, -1)

        # Threshold tick
        ty = sy + strip_h - int(THRESHOLD * strip_h)
        cv2.line(f, (sx - 2, ty), (w, ty), C['white'], 1)

        # Cooldown overlay (thin inner strip)
        if cooldown > 0:
            cd_h = int((cooldown / COOLDOWN_FRAMES) * strip_h)
            cv2.rectangle(f, (sx, sy), (sx + 3, sy + cd_h), C['orange'], -1)

    # ── Overlays ──────────────────────────────────────────────────────────────

    def _draw_paused(self, f, w, h):
        rr(f, w // 2 - 120, h // 2 - 24, w // 2 + 120, h // 2 + 24,
           C['orange'], r=12, alpha=0.93)
        txt(f, 'PAUSED  (P to resume)',
            (w // 2, h // 2 + 9), scale=0.62, color=C['bg'], bold=True, anchor='center')

    def _draw_switch_progress(self, f, w, h, mode, mc, switch_cnt):
        BAR    = 240
        cx, cy = w // 2, h // 2
        prog   = min(1.0, switch_cnt / SWITCH_FRAMES)
        target = 'MOUSE' if mode == 'GESTURES' else 'GESTURES'
        _semi(f, cx - BAR // 2 - 22, cy - 46, cx + BAR // 2 + 22, cy + 26,
              C['bg'], alpha=0.90)
        txt(f, f'HOLD  ->  {target}',
            (cx, cy - 14), scale=0.68, color=mc, bold=True, anchor='center')
        cv2.rectangle(f, (cx - BAR // 2, cy + 2),
                      (cx + BAR // 2, cy + 16), C['dark'], -1)
        if prog > 0:
            cv2.rectangle(f, (cx - BAR // 2, cy + 2),
                          (cx - BAR // 2 + int(prog * BAR), cy + 16), mc, -1)

    def _draw_volume(self, f, h, vol: int):
        TH, BH = self.TOP_H, self.BOT_H
        VH     = h - TH - BH - 20
        vx, vy = 12, TH + 10
        fill   = int(vol / 100 * VH)
        _rr_fill(f, vx, vy, vx + 16, vy + VH, C['dark'], r=4)
        if fill > 0:
            col = C['green'] if vol > 30 else C['orange']
            _rr_fill(f, vx, vy + VH - fill, vx + 16, vy + VH, col, r=4)
        txt(f, f'{vol}%',   (vx, vy - 10),         scale=0.42, color=C['green'])
        txt(f, 'VOL',       (vx - 2, vy + VH + 16), scale=0.36, color=C['gray'])

    def _draw_flash(self, f, w, h, mc):
        alpha      = min(1.0, (self._flash_end - time.time()) / 0.35)
        fw, fh     = tsize(self._flash_msg, 0.72, bold=True)
        fx         = w // 2 - fw // 2
        fy         = h // 2 + 74
        _semi(f, fx - 22, fy - fh - 14, fx + fw + 22, fy + 14,
              C['bg2'], alpha=alpha * 0.93)
        txt(f, self._flash_msg, (fx, fy), scale=0.72, color=mc, bold=True)

    def _draw_help(self, f, w, h):
        lines = [
            'H  help   |  T  test  |  P  pause',
            'M  menu   |  Q  quit  |  1-4 profile',
            'S  settings panel',
            '',
            'ok   ->  MOUSE mode',
            'stop ->  GESTURES mode',
            'left fist + right hand = VOLUME',
        ]
        pw = 294
        ph = len(lines) * 20 + 40
        px = w - pw - 10
        py = self.TOP_H + 8
        rr(f, px, py, px + pw, py + ph, C['bg2'], r=10, alpha=0.94)
        cv2.line(f, (px, py + 30), (px + pw, py + 30), C['accent'], 1)
        txt(f, 'SHORTCUTS', (px + pw // 2, py + 20),
            scale=0.44, color=C['accent'], bold=True, anchor='center')
        for i, line in enumerate(lines):
            c = C['gray'] if not line else C['white']
            txt(f, line, (px + 10, py + 40 + i * 20), scale=0.37, color=c)

    def _draw_test_panel(self, f, w, all_probs: np.ndarray):
        order  = np.argsort(all_probs)[::-1]
        n      = len(GESTURES)
        ROW    = 28
        PW     = 220
        PH     = n * ROW + 44
        px     = w - PW - 18
        py     = self.TOP_H + 8

        rr(f, px - 4, py, px + PW + 4, py + PH, C['bg2'], r=10, alpha=0.94)
        cv2.line(f, (px - 4, py + 32), (px + PW + 4, py + 32), C['orange'], 1)
        txt(f, 'PROBABILITIES', (px + PW // 2, py + 20),
            scale=0.44, color=C['orange'], bold=True, anchor='center')

        for row, gi in enumerate(order):
            g    = GESTURES[gi]
            prob = all_probs[gi]
            ry   = py + 38 + row * ROW
            BAR  = PW - 72
            col  = (C['green']  if prob >= THRESHOLD else
                    C['orange'] if prob >= 0.4       else C['dark'])
            txt(f, g.replace('_', ' '), (px, ry + 13),
                scale=0.36, color=C['white'] if prob >= 0.4 else C['gray'])
            cv2.rectangle(f, (px + 80, ry + 2), (px + 80 + BAR, ry + 18), C['dark'], -1)
            fill = int(prob * BAR)
            if fill > 0:
                cv2.rectangle(f, (px + 80, ry + 2),
                              (px + 80 + fill, ry + 18), col, -1)
            txt(f, f'{int(prob * 100)}%',
                (px + 80 + BAR + 4, ry + 14),
                scale=0.35, color=col if prob >= 0.4 else C['gray'])

    # ── Settings panel ────────────────────────────────────────────────────────

    def _draw_settings(self, f, w, h, cur_cam: int, cur_res: tuple):
        PW  = self.SETTINGS_PW
        px  = w - PW
        HDR = 46   # fixed header height

        # Left-side dim
        _semi(f, 0, 0, px, h, C['bg'], alpha=0.48)

        # Panel background
        _semi(f, px, 0, w, h, C['bg2'], alpha=0.97)
        cv2.line(f, (px, 0), (px, h), C['accent'], 2)

        # ── Fixed header (not scrollable) ─────────────────────────────────────
        cv2.rectangle(f, (px, 0), (w, HDR), C['bg3'], -1)
        cv2.line(f, (px, HDR), (w, HDR), C['accent'], 1)
        txt(f, 'SETTINGS & GUIDE',
            (px + (PW - 40) // 2, 28),
            scale=0.62, color=C['accent'], bold=True, anchor='center')

        # Red X close button
        xb_x1, xb_y1, xb_x2, xb_y2 = w - 38, 6, w - 8, 40
        rr(f, xb_x1, xb_y1, xb_x2, xb_y2, C['red'], r=6)
        mx, my = (xb_x1 + xb_x2) // 2, (xb_y1 + xb_y2) // 2
        cv2.line(f, (mx - 7, my - 7), (mx + 7, my + 7), C['white'], 2)
        cv2.line(f, (mx + 7, my - 7), (mx - 7, my + 7), C['white'], 2)
        self._x_rect = (xb_x1, xb_y1, xb_x2, xb_y2)

        # ── Scrollable content ────────────────────────────────────────────────
        # Clip drawing to content area [HDR, h] using sub-image
        content_h = h - HDR
        roi       = f[HDR:h, px:w]          # view into frame (no copy)

        self._s_btns = {}
        y  = 10 - self._scroll              # virtual y inside ROI
        MP = 12                             # left/right margin inside panel

        def visible(y1, y2):
            return y2 > 0 and y1 < content_h

        def sec(title):
            nonlocal y
            if visible(y, y + 22):
                cv2.putText(roi, title, (MP, y + 14),
                            _FONT, 0.42, C['gray'], 1, cv2.LINE_AA)
                cv2.line(roi, (MP, y + 20), (PW - MP, y + 20), C['bg3'], 1)
            y += 26

        def div():
            nonlocal y
            if visible(y, y + 1):
                cv2.line(roi, (MP, y), (PW - MP, y), C['border'], 1)
            y += 12

        hand_col = {'R': C['accent'], 'L': C['orange']}

        # ── CAMERA ───────────────────────────────────────────────────────────
        sec('CAMERA')
        row_w = PW - 2 * MP
        for cam_idx, cam_label in self._cameras:
            lbl      = cam_label[:24]   # truncate to prevent overflow
            _, th    = tsize(lbl, 0.43)
            y1, y2   = y, y + 28
            selected = cam_idx == cur_cam
            if visible(y1, y2):
                bg  = C['accent'] if selected else C['dark']
                col = C['bg']     if selected else C['white']
                _rr_fill(roi, MP, y1, MP + row_w, y2, bg, r=5)
                cv2.putText(roi, lbl, (MP + 9, y1 + th + 5),
                            _FONT, 0.43, col, 1, cv2.LINE_AA)
            self._s_btns[f'cam_{cam_idx}'] = (
                px + MP, HDR + y1, px + MP + row_w, HDR + y2)
            y += 32
        div()

        # ── DISPLAY RESOLUTION ────────────────────────────────────────────────
        sec('DISPLAY RESOLUTION')
        for i, (res, lbl, hint) in enumerate(self._resolutions):
            selected = res == cur_res
            tw, _  = tsize(lbl,  0.45)
            hw, _  = tsize(hint, 0.31)
            bw     = max(tw + 22, hw + 16)
            y1, y2 = y, y + 46
            if visible(y1, y2):
                bg   = C['accent'] if selected else C['dark']
                col  = C['bg']     if selected else C['white']
                col2 = C['bg']     if selected else C['gray']
                _rr_fill(roi, MP, y1, MP + bw, y2, bg, r=7)
                cv2.putText(roi, lbl,  (MP + bw // 2 - tw // 2, y1 + 18),
                            _FONT, 0.45, col,  2 if selected else 1, cv2.LINE_AA)
                cv2.putText(roi, hint, (MP + bw // 2 - hw // 2, y1 + 34),
                            _FONT, 0.31, col2, 1, cv2.LINE_AA)
            self._s_btns[f'res_{i}'] = (
                px + MP, HDR + y1, px + MP + bw, HDR + y2)
            y += 52
        div()

        # ── GESTURE GUIDE ─────────────────────────────────────────────────────
        sec('GESTURE GUIDE')
        for gname, hand, desc in GESTURE_GUIDE:
            y1, y2 = y, y + 44
            if visible(y1, y2):
                hcol = hand_col[hand]
                _rr_fill(roi, MP, y + 3, MP + 18, y + 19, hcol, r=4)
                cv2.putText(roi, hand, (MP + 4, y + 16),
                            _FONT, 0.36, C['bg'], 1, cv2.LINE_AA)
                cv2.putText(roi, gname.replace('_', ' '),
                            (MP + 24, y + 17),
                            _FONT, 0.44, C['white'], 1, cv2.LINE_AA)
                cv2.putText(roi, desc, (MP + 24, y + 34),
                            _FONT, 0.35, C['gray'], 1, cv2.LINE_AA)
            y += 46
        div()

        # ── KEYBOARD ──────────────────────────────────────────────────────────
        sec('KEYBOARD')
        for key, desc in SHORTCUTS:
            if visible(y, y + 22):
                kw, kh = tsize(key, 0.40, bold=True)
                _rr_fill(roi, MP, y + 1, MP + kw + 14, y + 18, C['bg3'], r=5)
                cv2.putText(roi, key,  (MP + 7,  y + 15),
                            _FONT, 0.40, C['accent'], 2, cv2.LINE_AA)
                cv2.putText(roi, desc, (MP + kw + 20, y + 15),
                            _FONT, 0.37, C['white'], 1, cv2.LINE_AA)
            y += 22
        div()

        # ── AUTO-PROFILE ──────────────────────────────────────────────────────
        sec('AUTO-PROFILE')
        auto_lines = [
            ('Chrome / Firefox / Edge  ->  browser', C['white']),
            ('VS Code / PyCharm        ->  vscode',  C['white']),
            ('Spotify / VLC            ->  media',   C['white']),
            ('Everything else          ->  default', C['white']),
            ('',                                     C['bg']),
            ('Keys 1-4  =  manual override',         C['yellow']),
        ]
        for line, col in auto_lines:
            if visible(y, y + 18) and line:
                cv2.putText(roi, line, (MP, y + 14),
                            _FONT, 0.36, col, 1, cv2.LINE_AA)
            y += 18

        # ── Scroll tracking & scrollbar ───────────────────────────────────────
        total_virtual = y + self._scroll  # total content height
        self._scroll_max = max(0, total_virtual - content_h + 20)

        if self._scroll_max > 0:
            sb_track = content_h
            thumb_h  = max(28, int(sb_track * content_h / total_virtual))
            thumb_y  = int(self._scroll / self._scroll_max *
                           (sb_track - thumb_h))
            cv2.rectangle(f, (w - 5, HDR),        (w, h),              C['dark'],   -1)
            cv2.rectangle(f, (w - 5, HDR+thumb_y), (w, HDR+thumb_y+thumb_h),
                          C['accent'], -1)
