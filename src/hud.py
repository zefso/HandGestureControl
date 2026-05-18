"""src/hud.py — HUD, settings panel, and draw helpers (Modernist Redesign)."""
import cv2
import numpy as np
import time

from src.config import (
    GESTURES, THRESHOLD, COOLDOWN_FRAMES, SWITCH_FRAMES,
)

# ── Color palette (Modernist) ────────────────────────────────────────────────
C: dict[str, tuple] = {
    'bg':      (15,  15,  18),
    'panel':   (32,  32,  35),
    'border':  (55,  55,  60),
    'accent':  (250, 160, 20),   # Cyan
    'green':   (100, 220, 80),
    'orange':  (20,  140, 255),
    'purple':  (230, 80,  180),
    'red':     (80,  60,  240),
    'white':   (245, 245, 250),
    'gray':    (140, 140, 150),
    'dark':    (40,  40,  45),
    'yellow':  (50,  220, 240),
}

MODE_COLORS: dict[str, tuple] = {
    'GESTURES': C['accent'],
    'MOUSE':    C['purple'],
    'VOLUME':   C['green'],
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX

GESTURE_GUIDE = [
    ('static',      'R', 'Idle / neutral — no action'),
    ('swipe_right', 'R', 'Swipe right hand to the right'),
    ('swipe_left',  'L', 'Swipe left hand to the left'),
    ('ok',          'R', 'Touch index + thumb → MOUSE mode'),
    ('stop',        'R', 'Open palm → back to GESTURES'),
    ('browser',     'R', 'Peace sign: index + middle up'),
    ('fist_left',   'L', 'Make a fist with left hand'),
    ('swipe_up',    'R', 'Swipe right hand upward'),
    ('swipe_down',  'R', 'Swipe right hand downward'),
    ('call_me',     'R', 'Pinky + thumb out (phone sign)'),
]

SHORTCUTS = [
    ('H',       'Help overlay on / off'),
    ('T',       'Test mode (no actions fired)'),
    ('P',       'Pause / Resume'),
    ('M',       'Quit'),
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
        cv2.circle(img, (cx, cy), r, color, 1, cv2.LINE_AA)
    
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, 1, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, 1, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, 1, cv2.LINE_AA)

def rr(img, x1, y1, x2, y2, color, r: int = 10, alpha: float = 1.0, shadow: bool = False):
    """Rounded rectangle with optional transparency and shadow."""
    if shadow:
        so = 3
        ov_sh = img.copy()
        _rr_fill(ov_sh, x1+so, y1+so, x2+so, y2+so, (0,0,0), r)
        cv2.addWeighted(ov_sh, 0.4 * alpha, img, 1 - 0.4 * alpha, 0, img)
    if alpha < 1.0:
        ov = img.copy()
        _rr_fill(ov, x1, y1, x2, y2, color, r)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    else:
        _rr_fill(img, x1, y1, x2, y2, color, r)

def txt(img, text: str, pos: tuple, scale: float = 0.52,
        color=None, bold: bool = False, anchor: str = 'left'):
    color  = color or C['white']
    weight = 1  # Force non-bold for readability
    if anchor != 'left':
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, weight)
        if anchor == 'center':
            pos = (pos[0] - tw // 2, pos[1] + th // 2)
        else:
            pos = (pos[0] - tw, pos[1])
    # Very subtle text shadow
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), _FONT, scale, (10,10,10), weight, cv2.LINE_AA)
    cv2.putText(img, text, pos, _FONT, scale, color, weight, cv2.LINE_AA)

def tsize(text: str, scale: float = 0.52, bold: bool = False) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, _FONT, scale, 1)
    return w, h

def draw_gear(img, cx: int, cy: int,
              r_out: int = 12, r_in: int = 7, n: int = 6,
              color=(200, 200, 210)):
    n4     = n * 4
    angles = [i * 2 * np.pi / n4 - np.pi / 2 for i in range(n4)]
    pts    = [[int(cx + (r_out if i % 4 < 2 else r_in) * np.cos(a)),
               int(cy + (r_out if i % 4 < 2 else r_in) * np.sin(a))]
              for i, a in enumerate(angles)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(1, r_in - 2), C['bg'], -1)
    cv2.circle(img, (cx, cy), max(1, r_in - 2), color, 1, cv2.LINE_AA)

# ── HUD ───────────────────────────────────────────────────────────────────────

class HUD:
    SETTINGS_PW = 360

    def __init__(self):
        self.help_on       = False
        self.settings_open = False
        self._flash_msg    = ''
        self._flash_end    = 0.0
        self._gear_rect    = (0, 0, 1, 1)
        self._x_rect       = (0, 0, 1, 1)
        self._s_btns: dict = {}
        self._scroll       = 0
        self._scroll_max   = 0
        self._cameras: list        = []
        self._resolutions: list    = []
        self.selected_cam: int | None   = None
        self.selected_res: tuple | None = None

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
        if self.is_gear_click(x, y):
            self.toggle_settings()
            return
        if not self.settings_open:
            return
        x1, y1, x2, y2 = self._x_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.settings_open = False
            return
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
        if self.settings_open:
            self._scroll = max(0, min(self._scroll + delta, self._scroll_max))

    def draw(self, frame: np.ndarray, *,
             mode: str, profile: str, gesture: str, confidence: float,
             last_action: str, history, fps: float,
             cooldown: int, switch_cnt: int, switching: bool,
             paused: bool = False, vol: int | None = None,
             test_mode: bool = False, all_probs: np.ndarray | None = None,
             auto_profile: bool = True,
             cameras: list | None = None, cur_cam: int = 0,
             resolutions: list | None = None, cur_res: tuple = (1280, 720)):

        self._cameras    = cameras or []
        self._resolutions = resolutions or []

        h, w = frame.shape[:2]
        mc   = MODE_COLORS.get(mode, C['accent'])

        # Floating elements layout
        self._draw_top_pills(frame, w, h, mode, mc, profile, gesture,
                             confidence, fps, test_mode, auto_profile)
        self._draw_bottom_pills(frame, w, h, last_action, history)
        self._draw_conf_strip(frame, w, h, confidence, cooldown, mode, mc)

        if paused:
            self._draw_paused(frame, w, h)
        if switching:
            self._draw_switch_progress(frame, w, h, mode, mc, switch_cnt)
        if vol is not None:
            self._draw_volume(frame, w, h, vol)
        if test_mode and all_probs is not None:
            self._draw_test_panel(frame, w, all_probs)
        if time.time() < self._flash_end:
            self._draw_flash(frame, w, h, mc)
        if self.settings_open:
            self._draw_settings(frame, w, h, cur_cam, cur_res)
        elif self.help_on:
            self._draw_help(frame, w, h)

    def _draw_top_pills(self, f, w, h, mode, mc, profile, gesture,
                        confidence, fps, test_mode, auto_profile):
        pad = 20
        # Mode Badge
        badge = f' {mode} '
        bw, bh = tsize(badge, 0.45, bold=True)
        rr(f, pad, pad, pad + bw + 24, pad + 38, C['panel'], r=19, alpha=0.85, shadow=True)
        # Colored dot
        cv2.circle(f, (pad + 16, pad + 19), 5, mc, -1)
        txt(f, badge, (pad + 24, pad + 25), scale=0.45, color=C['white'], bold=True)

        x_cur = pad + bw + 34

        # Action / Gesture
        if gesture not in ('static', '') and confidence > 0.5:
            gname = gesture.replace('_', ' ').upper()
            gw, gh = tsize(gname, 0.45, bold=True)
            rr(f, x_cur, pad, x_cur + gw + 24, pad + 38, C['panel'], r=19, alpha=0.85, shadow=True)
            txt(f, gname, (x_cur + 12, pad + 25), scale=0.45, color=mc, bold=True)
            x_cur += gw + 34

        # Profile
        ptw, _ = tsize(profile.upper(), 0.42, bold=True)
        rr(f, x_cur, pad, x_cur + ptw + 24, pad + 38, C['panel'], r=19, alpha=0.85, shadow=True)
        txt(f, profile.upper(), (x_cur + 12, pad + 25), scale=0.42, color=C['white'], bold=True)

        # Right side: Settings & FPS
        fps_str = f'{fps:.0f}'
        fw, _ = tsize(fps_str, 0.45, bold=True)
        gear_size = 38
        rx = w - pad - gear_size
        
        # Gear Button
        gcol = C['accent'] if self.settings_open else C['panel']
        rr(f, rx, pad, rx + gear_size, pad + gear_size, gcol, r=19, alpha=0.85, shadow=True)
        draw_gear(f, rx + 19, pad + 19, color=C['bg'] if self.settings_open else C['white'])
        self._gear_rect = (rx, pad, rx + gear_size, pad + gear_size)

        rx -= (fw + 30)
        # FPS Pill
        fps_col = C['green'] if fps >= 25 else C['orange'] if fps >= 15 else C['red']
        rr(f, rx, pad, rx + fw + 20, pad + 38, C['panel'], r=19, alpha=0.85, shadow=True)
        cv2.circle(f, (rx + 12, pad + 19), 4, fps_col, -1)
        txt(f, fps_str, (rx + 22, pad + 25), scale=0.45, color=C['white'], bold=True)
        
        # TEST mode pill
        if test_mode:
            tw, _ = tsize('TEST', 0.45, bold=True)
            rx -= (tw + 30)
            rr(f, rx, pad, rx + tw + 20, pad + 38, C['orange'], r=19, alpha=0.9, shadow=True)
            txt(f, 'TEST', (rx + 10, pad + 25), scale=0.45, color=C['bg'], bold=True)

    def _draw_bottom_pills(self, f, w, h, last_action, history):
        pad = 20
        # Bottom Left: Last Action
        aw, _ = tsize(last_action, 0.48)
        rr(f, pad, h - pad - 42, pad + aw + 40, h - pad, C['panel'], r=21, alpha=0.85, shadow=True)
        cv2.putText(f, '>', (pad + 16, h - pad - 14), _FONT, 0.48, C['accent'], 2, cv2.LINE_AA)
        txt(f, last_action, (pad + 34, h - pad - 15), scale=0.48, color=C['white'])

        # Bottom Right: History (Chips)
        chips = list(history)[-4:]
        cx = w - pad - 24 # leave space for confidence strip
        for g in reversed(chips):
            tw, _ = tsize(g, 0.35)
            chip_w = tw + 20
            cx -= chip_w
            rr(f, cx, h - pad - 34, cx + chip_w, h - pad - 8, C['dark'], r=13, alpha=0.8)
            txt(f, g, (cx + 10, h - pad - 17), scale=0.35, color=C['gray'])
            cx -= 8

    def _draw_conf_strip(self, f, w, h, confidence, cooldown, mode, mc):
        if mode != 'GESTURES':
            return
        pad = 20
        strip_h = h - 2 * pad - 90
        sx = w - pad - 6
        sy = pad + 45
        
        # Background track
        rr(f, sx, sy, sx + 6, sy + strip_h, C['panel'], r=3, alpha=0.6)
        
        col  = C['green'] if confidence >= THRESHOLD else C['orange'] if confidence >= 0.6 else C['gray']
        fill = int(confidence * strip_h)
        if fill > 0:
            rr(f, sx, sy + strip_h - fill, sx + 6, sy + strip_h, col, r=3, alpha=0.9)

        # Threshold tick
        ty = sy + strip_h - int(THRESHOLD * strip_h)
        cv2.line(f, (sx - 2, ty), (sx + 8, ty), C['white'], 2)

        # Cooldown overly
        if cooldown > 0:
            cd_h = int((cooldown / COOLDOWN_FRAMES) * strip_h)
            rr(f, sx, sy, sx + 6, sy + cd_h, C['orange'], r=3, alpha=0.9)

    def _draw_paused(self, f, w, h):
        rr(f, w // 2 - 140, h // 2 - 35, w // 2 + 140, h // 2 + 35, C['orange'], r=18, alpha=0.85, shadow=True)
        txt(f, 'PAUSED', (w // 2, h // 2 - 2), scale=0.7, color=C['bg'], bold=True, anchor='center')
        txt(f, 'Press P to resume', (w // 2, h // 2 + 20), scale=0.4, color=C['bg'], anchor='center')

    def _draw_switch_progress(self, f, w, h, mode, mc, switch_cnt):
        BAR = 280
        cx, cy = w // 2, h // 2
        prog = min(1.0, switch_cnt / SWITCH_FRAMES)
        target = 'MOUSE' if mode == 'GESTURES' else 'GESTURES'
        
        rr(f, cx - BAR // 2 - 25, cy - 50, cx + BAR // 2 + 25, cy + 30, C['panel'], r=18, alpha=0.9, shadow=True)
        txt(f, f'Switching to {target}', (cx, cy - 15), scale=0.55, color=C['white'], bold=True, anchor='center')
        
        # Track
        rr(f, cx - BAR // 2, cy + 8, cx + BAR // 2, cy + 14, C['dark'], r=3)
        # Fill
        if prog > 0:
            rr(f, cx - BAR // 2, cy + 8, cx - BAR // 2 + int(prog * BAR), cy + 14, mc, r=3)

    def _draw_volume(self, f, w, h, vol: int):
        pad = 20
        VH = h - 150
        vx, vy = pad, pad + 65
        fill = int(vol / 100 * VH)
        
        rr(f, vx, vy, vx + 16, vy + VH, C['panel'], r=8, alpha=0.85, shadow=True)
        if fill > 0:
            col = C['green'] if vol > 30 else C['orange']
            rr(f, vx, vy + VH - fill, vx + 16, vy + VH, col, r=8, alpha=0.9)
            
        txt(f, f'{vol}%', (vx + 8, vy - 12), scale=0.45, color=C['white'], bold=True, anchor='center')

    def _draw_flash(self, f, w, h, mc):
        alpha = min(1.0, max(0.0, (self._flash_end - time.time()) / 0.35))
        if alpha <= 0: return
        fw, fh = tsize(self._flash_msg, 0.65, bold=True)
        fx, fy = w // 2, h - 130
        rr(f, fx - fw//2 - 25, fy - fh - 18, fx + fw//2 + 25, fy + 18, C['panel'], r=20, alpha=alpha * 0.9, shadow=True)
        txt(f, self._flash_msg, (fx, fy), scale=0.65, color=mc, bold=True, anchor='center')

    def _draw_help(self, f, w, h):
        pw = 340
        ph = len(SHORTCUTS) * 28 + 70
        px = w - pw - 20
        py = 80
        rr(f, px, py, px + pw, py + ph, C['panel'], r=18, alpha=0.92, shadow=True)
        txt(f, 'SHORTCUTS', (px + pw // 2, py + 30), scale=0.5, color=C['accent'], bold=True, anchor='center')
        cv2.line(f, (px + 20, py + 48), (px + pw - 20, py + 48), C['border'], 1)
        
        for i, (key, desc) in enumerate(SHORTCUTS):
            ry = py + 75 + i * 28
            kw, _ = tsize(key, 0.4, bold=True)
            rr(f, px + 20, ry - 15, px + 20 + kw + 16, ry + 5, C['dark'], r=6)
            txt(f, key, (px + 28, ry), scale=0.4, color=C['accent'], bold=True)
            txt(f, desc, (px + 45 + kw, ry), scale=0.4, color=C['white'])

    def _draw_test_panel(self, f, w, all_probs: np.ndarray):
        order  = np.argsort(all_probs)[::-1]
        PW     = 260
        PH     = len(GESTURES) * 30 + 60
        px     = w - PW - 20
        py     = 80

        rr(f, px, py, px + PW, py + PH, C['panel'], r=18, alpha=0.92, shadow=True)
        txt(f, 'PROBABILITIES', (px + PW // 2, py + 30), scale=0.5, color=C['orange'], bold=True, anchor='center')
        cv2.line(f, (px + 20, py + 48), (px + PW - 20, py + 48), C['border'], 1)

        for row, gi in enumerate(order):
            g    = GESTURES[gi]
            prob = all_probs[gi]
            ry   = py + 75 + row * 30
            BAR  = PW - 120
            col  = C['green'] if prob >= THRESHOLD else C['orange'] if prob >= 0.4 else C['dark']
            txt(f, g.replace('_', ' '), (px + 20, ry), scale=0.4, color=C['white'] if prob >= 0.4 else C['gray'])
            rr(f, px + 110, ry - 8, px + 110 + BAR, ry + 2, C['dark'], r=4)
            fill = int(prob * BAR)
            if fill > 0:
                rr(f, px + 110, ry - 8, px + 110 + fill, ry + 2, col, r=4)
            txt(f, f'{int(prob * 100)}%', (px + 115 + BAR, ry), scale=0.35, color=col if prob >= 0.4 else C['gray'])

    def _draw_settings(self, f, w, h, cur_cam: int, cur_res: tuple):
        PW  = self.SETTINGS_PW
        px  = w - PW
        HDR = 75
        
        # Dim background
        ov = f.copy()
        cv2.rectangle(ov, (0, 0), (px, h), C['bg'], -1)
        cv2.addWeighted(ov, 0.4, f, 0.6, 0, f)

        # Panel
        cv2.rectangle(f, (px, 0), (w, h), C['panel'], -1)
        cv2.line(f, (px, 0), (px, h), C['border'], 1)

        # Header
        txt(f, 'SETTINGS', (px + 30, 48), scale=0.6, color=C['white'], bold=True)
        # Close button
        xb_x1, xb_y1, xb_x2, xb_y2 = w - 50, 25, w - 20, 55
        rr(f, xb_x1, xb_y1, xb_x2, xb_y2, C['dark'], r=15)
        mx, my = (xb_x1 + xb_x2) // 2, (xb_y1 + xb_y2) // 2
        cv2.line(f, (mx - 6, my - 6), (mx + 6, my + 6), C['gray'], 2, cv2.LINE_AA)
        cv2.line(f, (mx + 6, my - 6), (mx - 6, my + 6), C['gray'], 2, cv2.LINE_AA)
        self._x_rect = (xb_x1, xb_y1, xb_x2, xb_y2)
        cv2.line(f, (px + 20, HDR), (w - 20, HDR), C['border'], 1)

        content_h = h - HDR
        roi = f[HDR:h, px:w]
        self._s_btns = {}
        y = 15 - self._scroll
        MP = 25

        def visible(y1, y2): return y2 > 0 and y1 < content_h
        def sec(title):
            nonlocal y
            if visible(y, y + 30):
                txt(roi, title, (MP, y + 20), scale=0.45, color=C['accent'], bold=True)
            y += 35

        # CAMERAS
        sec('Camera')
        row_w = PW - 2 * MP
        for cam_idx, cam_label in self._cameras:
            lbl = cam_label[:24]
            y1, y2 = y, y + 40
            selected = cam_idx == cur_cam
            if visible(y1, y2):
                bg = C['accent'] if selected else C['dark']
                col = C['bg'] if selected else C['white']
                rr(roi, MP, y1, MP + row_w, y2, bg, r=10)
                txt(roi, lbl, (MP + 15, y1 + 25), scale=0.45, color=col, bold=selected)
            self._s_btns[f'cam_{cam_idx}'] = (px + MP, HDR + y1, px + MP + row_w, HDR + y2)
            y += 48
        y += 10

        # RESOLUTION
        sec('Resolution')
        for i, (res, lbl, hint) in enumerate(self._resolutions):
            selected = res == cur_res
            tw, _ = tsize(lbl, 0.45, bold=selected)
            hw, _ = tsize(hint, 0.35)
            bw = max(tw + 30, hw + 20)
            y1, y2 = y, y + 60
            if visible(y1, y2):
                bg = C['accent'] if selected else C['dark']
                col = C['bg'] if selected else C['white']
                col2 = C['bg'] if selected else C['gray']
                rr(roi, MP, y1, MP + bw, y2, bg, r=12)
                txt(roi, lbl, (MP + bw//2, y1 + 26), scale=0.45, color=col, bold=selected, anchor='center')
                txt(roi, hint, (MP + bw//2, y1 + 46), scale=0.35, color=col2, anchor='center')
            self._s_btns[f'res_{i}'] = (px + MP, HDR + y1, px + MP + bw, HDR + y2)
            y += 70
        y += 10

        # GESTURES
        sec('Gesture Guide')
        for gname, hand, desc in GESTURE_GUIDE:
            y1, y2 = y, y + 50
            if visible(y1, y2):
                hcol = C['accent'] if hand == 'R' else C['orange']
                rr(roi, MP, y + 8, MP + 26, y + 34, hcol, r=8)
                txt(roi, hand, (MP + 13, y + 26), scale=0.45, color=C['bg'], bold=True, anchor='center')
                txt(roi, gname.replace('_', ' '), (MP + 40, y + 24), scale=0.45, color=C['white'], bold=True)
                txt(roi, desc, (MP + 40, y + 44), scale=0.35, color=C['gray'])
            y += 55
        y += 10

        # SCROLLBAR
        total_virtual = y + self._scroll
        self._scroll_max = max(0, total_virtual - content_h + 20)
        if self._scroll_max > 0:
            sb_track = content_h - 30
            thumb_h = max(40, int(sb_track * content_h / total_virtual))
            thumb_y = int(self._scroll / self._scroll_max * (sb_track - thumb_h)) + 15
            rr(f, w - 8, HDR + thumb_y, w - 4, HDR + thumb_y + thumb_h, C['gray'], r=2, alpha=0.5)
