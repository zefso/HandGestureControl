"""src/app.py — GestureControlApp and GestureActionExecutor."""
import cv2
import torch
import numpy as np
import pyautogui
import sys
import os
import time
import json
import webbrowser
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import mediapipe as mp
    from src.model   import GestureLSTM
    from src.utils   import extract_keypoints, VolumeController, reset_delta_state
    from src.config  import (
        GESTURES, GESTURE_ACTIONS,
        SEQ_LENGTH, MODEL_PATH, THRESHOLD, DEVICE,
        SWITCH_FRAMES, COOLDOWN_FRAMES, SWIPE_COOLDOWN_FRAMES,
        MOUSE_SMOOTHING,
    )
    from src.mouse_controller import AirMouse
    from src.hud import HUD, C, rr, txt, MODE_COLORS
except ImportError as e:
    print(f'[Import error] {e}')
    sys.exit(1)

pyautogui.PAUSE    = 0
pyautogui.FAILSAFE = False

# Exported for tests
SWIPE_GESTURES = ['swipe_left', 'swipe_right']
APP_VERSION    = '2.1'

# Available display resolutions shown in settings panel
DISP_RESOLUTIONS = [
    ((640,  480),  '640 x 480',   'native / fastest'),
    ((1280, 720),  '1280 x 720',  'HD'),
    ((1920, 1080), '1920 x 1080', 'Full HD'),
]


def _get_camera_labels() -> list[str]:
    """Return DirectShow device names in capture order (matches OpenCV indices)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        pass
    try:
        import win32com.client
        wmi = win32com.client.GetObject("winmgmts:")
        names = [item.Name for item in wmi.InstancesOf("Win32_PnPEntity")
                 if hasattr(item, 'PNPClass') and item.PNPClass in ('Camera', 'Image')
                 and item.Name]
        if names:
            return names
    except Exception:
        pass
    return []


def _detect_cameras(max_idx: int = 5) -> list[tuple[int, str]]:
    """Return available cameras as (index, display_name) pairs."""
    labels = _get_camera_labels()
    found  = []
    for i in range(max_idx):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened() and cap.read()[0]:
                name = labels[i] if i < len(labels) else f'Cam {i}'
                found.append((i, name))
            cap.release()
        except Exception:
            pass
    return found or [(0, 'Cam 0')]

# ── Auto-profile ───────────────────────────────────────────────────────────────

try:
    import win32gui
    _WIN32 = True
except ImportError:
    _WIN32 = False

_PROFILE_RULES = [
    (['chrome', 'firefox', 'edge', 'opera', 'brave', 'google'],           'browser'),
    (['code', 'vscode', 'visual studio', 'pycharm', 'intellij', 'atom'],  'vscode'),
    (['spotify', 'vlc', 'media player', 'foobar', 'aimp', 'winamp',
      'youtube', 'netflix'],                                               'media'),
]


def _foreground_title() -> str:
    if not _WIN32:
        return ''
    try:
        return win32gui.GetWindowText(win32gui.GetForegroundWindow()).lower()
    except Exception:
        return ''


def _title_to_profile(title: str) -> str:
    for keywords, profile in _PROFILE_RULES:
        if any(kw in title for kw in keywords):
            return profile
    return 'default'


# ── GestureActionExecutor ─────────────────────────────────────────────────────

class GestureActionExecutor:
    """Executes gesture actions from the active profile in gestures.json."""

    def __init__(self, profile: str = 'default'):
        self._profile = profile
        self._actions: dict = {}
        self._reload()

    def set_profile(self, profile: str):
        self._profile = profile
        self._reload()

    def _reload(self):
        try:
            path = os.path.join(os.path.dirname(__file__), 'gestures.json')
            with open(path, encoding='utf-8') as fp:
                cfg = json.load(fp)
            self._actions = (cfg.get('profiles', {})
                               .get(self._profile, {})
                               .get('actions', {}))
        except Exception:
            self._actions = GESTURE_ACTIONS

    def execute(self, gesture: str) -> str:
        a = self._actions.get(gesture)
        if not a:
            return gesture
        try:
            t = a.get('type')
            if t == 'hotkey':
                pyautogui.hotkey(*a.get('keys', []))
            elif t == 'key':
                pyautogui.press(a.get('key', ''))
            elif t == 'url':
                webbrowser.open(a.get('url', ''))
            elif t == 'command':
                os.system(a.get('cmd', ''))
        except Exception as e:
            print(f'[Executor] {e}')
        return a.get('description', gesture)


# ── GestureControlApp ─────────────────────────────────────────────────────────

class GestureControlApp:
    """
    Main inference loop.

    Modes:
      GESTURES — LSTM gesture recognition → actions
      MOUSE    — AirMouse controls cursor with right hand
      VOLUME   — left fist held + right hand controls volume

    Mode switch: ok (GESTURES→MOUSE) | stop (MOUSE→GESTURES)
    Auto-profile: detects the active window every 60 frames.
    """

    PROFILES        = ['default', 'browser', 'media', 'vscode']
    _AUTO_INTERVAL  = 60    # frames between auto-profile checks
    _FONT           = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, cam_idx: int = 0, disp_w: int = 1280, disp_h: int = 720):
        self._cam_idx  = cam_idx
        self._disp_w   = disp_w
        self._disp_h   = disp_h
        self._cameras  = _detect_cameras()
        self._profile = 'default'

        # Runtime state
        self._mode       = 'GESTURES'
        self._seq: list  = []
        self._cooldown   = 0
        self._swipe_cd   = 0
        self._fist_cnt   = 0
        self._switch_cnt = 0
        self._last_act   = 'ready'
        self._history    = deque(maxlen=12)
        self._fps        = 0.0
        self._fps_t      = time.time()
        self._paused     = False
        self._test_mode  = False
        self._confirm_buf = deque(maxlen=5)
        self._all_probs: np.ndarray | None = None

        # Auto-profile
        self._auto_profile     = True
        self._auto_cnt         = 0

        # Components
        self._executor = GestureActionExecutor(self._profile)
        self._hud      = HUD()
        self._mouse    = AirMouse(smoothing=MOUSE_SMOOTHING)
        self._volume   = VolumeController()
        self._model    = self._init_model()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_model(self) -> GestureLSTM:
        if not os.path.exists(MODEL_PATH):
            print(f'[ERROR] Model not found: {MODEL_PATH}')
            sys.exit(1)
        m = GestureLSTM(num_classes=len(GESTURES)).to(DEVICE)
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
        m.load_state_dict(state)
        m.eval()
        print(f'[OK] Model loaded | {len(GESTURES)} gestures | auto-profile ON')
        return m

    # ── Inference ─────────────────────────────────────────────────────────────

    def _predict(self, seq) -> tuple[str, float]:
        inp = torch.tensor(np.array([seq]), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self._model(inp), dim=1)
            prob, idx = torch.max(probs, 1)
        conf    = prob.item()
        gesture = GESTURES[idx.item()] if conf > THRESHOLD else 'static'
        return gesture, conf

    def _predict_all(self, seq) -> tuple[str, float, np.ndarray]:
        """Returns (gesture, confidence, all_probs) — used by TEST mode."""
        inp = torch.tensor(np.array([seq]), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            raw    = self._model(inp)
            all_p  = torch.softmax(raw, dim=1).cpu().numpy()[0]
        idx  = int(np.argmax(all_p))
        conf = float(all_p[idx])
        gesture = GESTURES[idx] if conf > THRESHOLD else 'static'
        return gesture, conf, all_p

    # ── Hand processing ───────────────────────────────────────────────────────

    def _process_hands(self, results):
        right, left, left_fist = None, None, False
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, info in enumerate(results.multi_handedness):
                label = info.classification[0].label
                lms   = results.multi_hand_landmarks[i]
                if label == 'Left':
                    left   = lms
                    closed = all(lms.landmark[j].y > lms.landmark[j - 2].y
                                 for j in [8, 12, 16, 20])
                    if closed and lms.landmark[0].y < 0.7:
                        left_fist = True
                else:
                    right = lms
        return right, left, left_fist

    # ── FPS ───────────────────────────────────────────────────────────────────

    def _tick_fps(self) -> float:
        now        = time.time()
        dt         = now - self._fps_t
        self._fps_t = now
        inst       = 1.0 / dt if dt > 1e-6 else 0.0
        self._fps  = 0.88 * self._fps + 0.12 * inst
        return self._fps

    # ── Auto-profile ──────────────────────────────────────────────────────────

    def _maybe_update_profile(self):
        if not self._auto_profile:
            return
        self._auto_cnt += 1
        if self._auto_cnt < self._AUTO_INTERVAL:
            return
        self._auto_cnt = 0
        title = _foreground_title()
        if title and 'gesture control' not in title:
            new = _title_to_profile(title)
            if new != self._profile:
                self._profile = new
                self._executor.set_profile(new)
                self._hud.flash(f'Auto: {new.upper()}')

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _handle_key(self, k: int) -> bool:
        """Returns False to exit run()."""
        if k in (ord('q'), 27):
            return False
        if k == ord('h'):
            self._hud.toggle_help()
        if k == ord('s'):
            self._hud.toggle_settings()
        if k == ord('p'):
            self._paused = not self._paused
            self._hud.flash('PAUSED' if self._paused else 'RESUMED')
        if k == ord('t'):
            self._test_mode = not self._test_mode
            self._confirm_buf.clear()
            self._hud.flash('TEST MODE ON  —  no actions' if self._test_mode
                            else 'TEST MODE OFF')
        if k == ord('m'):
            return False
        for i, p in enumerate(self.PROFILES):
            if k == ord(str(i + 1)):
                self._profile      = p
                self._auto_profile = False
                self._executor.set_profile(p)
                self._hud.flash(f'Manual: {p.upper()}')
        return True

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _mouse_cb(self, ev, x, y, flags, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self._hud.handle_click(x, y)
        elif ev == cv2.EVENT_MOUSEWHEEL:
            self._hud.handle_scroll(40 if flags < 0 else -40)

    # ── Gesture helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _verify_gesture(gesture: str, right_lms, left_lms=None) -> bool:
        """
        Geometric verification — second layer of defence after LSTM.
        Left-hand gestures use left_lms; right-hand gestures use right_lms.
        """
        LEFT_GESTURES = ('fist_left', 'swipe_left')
        lms = left_lms if gesture in LEFT_GESTURES else right_lms
        if lms is None:
            return True

        lm = lms.landmark

        if gesture == 'fist_left':
            return all(lm[tip].y > lm[tip - 2].y for tip in [8, 12, 16, 20])

        elif gesture == 'browser':
            return (lm[8].y  < lm[6].y  and   # index up
                    lm[12].y < lm[10].y and   # middle up
                    lm[16].y > lm[14].y and   # ring down
                    lm[20].y > lm[18].y)      # pinky down

        elif gesture == 'thumbs_up':
            return (lm[4].y < lm[3].y < lm[2].y and
                    all(lm[tip].y > lm[tip - 2].y for tip in [8, 12, 16, 20]))

        return True

    @staticmethod
    def _detect_thumbs_up(right_lms) -> bool:
        """Geometric-only thumbs-up detection (no LSTM needed)."""
        if right_lms is None:
            return False
        lm = right_lms.landmark
        return (lm[4].y < lm[3].y < lm[2].y < lm[1].y and
                lm[4].y < lm[9].y and
                all(lm[tip].y > lm[tip - 2].y for tip in [8, 12, 16, 20]))

    def _run_gesture(self, detected: str, right_lms=None, left_lms=None):
        is_swipe = detected in SWIPE_GESTURES
        can_fire = (self._swipe_cd == 0) if is_swipe else (self._cooldown == 0)

        if detected not in ('static', 'ok', 'stop') and can_fire:
            if not self._verify_gesture(detected, right_lms, left_lms):
                return
            desc = self._executor.execute(detected)
            self._last_act = desc
            self._history.append(detected)
            if is_swipe:
                self._swipe_cd = SWIPE_COOLDOWN_FRAMES
            else:
                self._cooldown = COOLDOWN_FRAMES
            print(f'>>> {detected.upper()} -> {desc}')

    # ── Skeleton drawing ──────────────────────────────────────────────────────

    def _draw_skeletons(self, frame, results, mp_h, mp_draw, mp_sty):
        if not results.multi_hand_landmarks:
            return
        for lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, lms,
                mp_h.HAND_CONNECTIONS,
                mp_sty.get_default_hand_landmarks_style(),
                mp_sty.get_default_hand_connections_style(),
            )

    # ── Mouse mode rendering ──────────────────────────────────────────────────

    def _run_mouse(self, frame, right):
        try:
            hp, wp  = frame.shape[:2]
            self._mouse.move(right)
            state = self._mouse.handle_actions(right)
            ix    = int(right.landmark[8].x * wp)
            iy    = int(right.landmark[8].y * hp)
            if state == 'L_DOWN':
                cv2.circle(frame, (ix, iy), 20, C['green'], -1)
                cv2.circle(frame, (ix, iy), 22, C['white'], 2)
            elif state == 'RIGHT_CLICK':
                cv2.circle(frame, (ix, iy), 20, C['red'], -1)
                cv2.circle(frame, (ix, iy), 22, C['white'], 2)
            elif state == 'SCROLLING':
                cv2.arrowedLine(frame, (ix, iy - 28), (ix, iy + 28), C['accent'], 3)
            else:
                cv2.circle(frame, (ix, iy), 10, C['purple'], 2)
        except Exception as e:
            print(f'[Mouse] {e}')

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _hud_draw(self, frame, *, fps, confirmed='static', conf=0.0,
                  switching=False, paused=False, vol=None,
                  DISP_W=1280, DISP_H=720):
        """Convenience wrapper — forwards all HUD.draw() params in one place."""
        self._hud.draw(
            frame,
            mode=self._mode, profile=self._profile,
            gesture=confirmed, confidence=conf,
            last_action=self._last_act,
            history=self._history, fps=fps,
            cooldown=self._cooldown,
            switch_cnt=self._switch_cnt, switching=switching,
            paused=paused, vol=vol,
            test_mode=self._test_mode,
            all_probs=self._all_probs,
            auto_profile=self._auto_profile,
            cameras=self._cameras, cur_cam=self._cam_idx,
            resolutions=DISP_RESOLUTIONS, cur_res=(DISP_W, DISP_H),
        )

    def run(self) -> None:
        mp_h    = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        mp_sty  = mp.solutions.drawing_styles

        cv2.namedWindow('Gesture Control Hub', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gesture Control Hub', self._disp_w, self._disp_h)
        cv2.setMouseCallback('Gesture Control Hub', self._mouse_cb)

        DISP_W, DISP_H = self._disp_w, self._disp_h

        with mp_h.Hands(max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:

            while True:   # outer restart loop (camera change)
                cap = cv2.VideoCapture(self._cam_idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print(f'[ERROR] Cannot open camera {self._cam_idx}')
                    break

                cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f'[Camera {self._cam_idx}] {cam_w}x{cam_h}  display {DISP_W}x{DISP_H}')
                need_scale = (cam_w != DISP_W or cam_h != DISP_H)

                cam_restart = False

                while cap.isOpened():
                    if cv2.getWindowProperty('Gesture Control Hub',
                                             cv2.WND_PROP_VISIBLE) < 1:
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    if need_scale:
                        frame = cv2.resize(frame, (DISP_W, DISP_H),
                                           interpolation=cv2.INTER_LINEAR)
                    fps = self._tick_fps()
                    k   = cv2.waitKey(1) & 0xFF

                    if not self._handle_key(k):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    self._maybe_update_profile()

                    # Apply settings changes from HUD
                    if self._hud.selected_cam is not None:
                        new_cam = self._hud.selected_cam
                        self._hud.selected_cam = None
                        if new_cam != self._cam_idx:
                            self._cam_idx = new_cam
                            cam_restart   = True
                            break
                    if self._hud.selected_res is not None:
                        DISP_W, DISP_H = self._hud.selected_res
                        self._disp_w, self._disp_h = DISP_W, DISP_H
                        need_scale = (cam_w != DISP_W or cam_h != DISP_H)
                        self._hud.selected_res = None
                        cv2.resizeWindow('Gesture Control Hub', DISP_W, DISP_H)
                        self._hud.flash(f'{DISP_W}x{DISP_H}')

                    # ── Pause ─────────────────────────────────────────────
                    if self._paused:
                        self._hud_draw(frame, fps=fps, paused=True,
                                       DISP_W=DISP_W, DISP_H=DISP_H)
                        cv2.imshow('Gesture Control Hub', frame)
                        continue

                    # ── MediaPipe ─────────────────────────────────────────
                    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    right, left, left_fist = self._process_hands(results)

                    detected  = 'static'
                    conf      = 0.0
                    switching = False

                    # ── VOLUME mode ───────────────────────────────────────
                    if left_fist:
                        self._fist_cnt += 1
                        if self._fist_cnt >= 7 and right:
                            vol = self._volume.apply(right)
                            self._last_act = f'Volume {vol}%'
                            self._draw_skeletons(frame, results, mp_h, mp_draw, mp_sty)
                            self._hud_draw(frame, fps=fps, vol=vol,
                                           DISP_W=DISP_W, DISP_H=DISP_H)
                            cv2.imshow('Gesture Control Hub', frame)
                            continue
                    else:
                        self._fist_cnt = 0

                    # ── Prediction ────────────────────────────────────────
                    self._all_probs = None
                    if results.multi_hand_landmarks:
                        kp = extract_keypoints(results)
                        self._seq.append(kp)
                        self._seq = self._seq[-SEQ_LENGTH:]
                        if len(self._seq) == SEQ_LENGTH:
                            if self._test_mode:
                                detected, conf, self._all_probs = self._predict_all(self._seq)
                            else:
                                detected, conf = self._predict(self._seq)

                    self._confirm_buf.append(detected)
                    confirmed = (detected
                                 if self._confirm_buf.count(detected) >= 3
                                 else 'static')

                    # ── Mode switch ───────────────────────────────────────
                    trigger = 'ok'    if self._mode == 'GESTURES' else 'stop'
                    target  = 'MOUSE' if self._mode == 'GESTURES' else 'GESTURES'
                    if confirmed == trigger:
                        self._switch_cnt += 1
                        switching = True
                        if self._switch_cnt > SWITCH_FRAMES:
                            self._mode       = target
                            self._switch_cnt = 0
                            self._seq        = []
                            self._confirm_buf.clear()
                            reset_delta_state()
                            self._hud.flash(f'-> {target} MODE')
                            switching = False
                    else:
                        self._switch_cnt = max(0, self._switch_cnt - 1)

                    # ── MOUSE mode ────────────────────────────────────────
                    if self._mode == 'MOUSE' and not switching and right:
                        self._run_mouse(frame, right)
                        self._last_act = 'MOUSE ACTIVE'

                    # ── GESTURES mode ─────────────────────────────────────
                    elif self._mode == 'GESTURES' and not switching:
                        geo = confirmed
                        if confirmed == 'static' and self._detect_thumbs_up(right):
                            geo  = 'thumbs_up'
                            conf = 1.0
                        if not self._test_mode:
                            self._run_gesture(geo, right, left)

                    # ── Cooldowns ─────────────────────────────────────────
                    if self._cooldown > 0: self._cooldown -= 1
                    if self._swipe_cd  > 0: self._swipe_cd  -= 1

                    # ── Render ────────────────────────────────────────────
                    self._draw_skeletons(frame, results, mp_h, mp_draw, mp_sty)
                    self._hud_draw(frame, fps=fps, confirmed=confirmed, conf=conf,
                                   switching=switching, DISP_W=DISP_W, DISP_H=DISP_H)
                    cv2.imshow('Gesture Control Hub', frame)

                cap.release()
                if not cam_restart:
                    break

        cv2.destroyAllWindows()
