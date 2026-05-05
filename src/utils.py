import numpy as np
from src.config import VOLUME_DISTANCE_MIN, VOLUME_DISTANCE_MAX


class VolumeController:
    """
    Керування гучністю Windows через pycaw.
    Гучність визначається відстанню між великим і вказівним пальцями.

    Діапазон відстані береться з config.py (VOLUME_DISTANCE_MIN/MAX).
    """

    def __init__(self, smoothing: float | None = None):
        from src.config import VOLUME_SMOOTHING
        self.smoothing = smoothing if smoothing is not None else VOLUME_SMOOTHING
        self._last_vol: int = 0
        self._interface = None

    def get_interface(self):
        """Lazy-ініціалізація COM-інтерфейсу аудіо (тільки при першому виклику)."""
        if self._interface is not None:
            return self._interface
        try:
            import comtypes
            from comtypes import CLSCTX_ALL, GUID
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IMMDeviceEnumerator
            from ctypes import cast, POINTER
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._interface = cast(interface, POINTER(IAudioEndpointVolume))
            except Exception:
                CLSID_MMDeviceEnumerator = GUID('{BCDE0395-E52F-467C-8E3D-C4579291692E}')
                enumerator = comtypes.CoCreateInstance(
                    CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, comtypes.CLSCTX_INPROC_SERVER
                )
                endpoint = enumerator.GetDefaultAudioEndpoint(0, 0)
                interface = endpoint.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._interface = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            print(f"[VolumeController] Критична помилка аудіо-інтерфейсу: {e}")
        return self._interface

    def calculate_level(self, hand_landmarks) -> int:
        """
        Повертає рівень гучності [0..100] за відстанню між пальцями.
        Застосовує dead-zone (±1.5%) і EMA-згладжування.
        """
        if not hand_landmarks:
            return self._last_vol

        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        distance = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5

        current_vol = float(np.interp(
            distance,
            [VOLUME_DISTANCE_MIN, VOLUME_DISTANCE_MAX],
            [0, 100]
        ))

        # Snap до меж
        if current_vol >= 97:
            self._last_vol = 100
            return 100
        if current_vol <= 3:
            self._last_vol = 0
            return 0

        # Dead-zone: ігноруємо мікрорухи
        if abs(current_vol - self._last_vol) < 1.5:
            return self._last_vol

        # EMA-згладжування
        smoothed = int(self.smoothing * current_vol + (1 - self.smoothing) * self._last_vol)
        self._last_vol = smoothed
        return smoothed

    def apply(self, hand_landmarks) -> int:
        """Розраховує рівень і застосовує до системної гучності."""
        level = self.calculate_level(hand_landmarks)
        interface = self.get_interface()
        if interface:
            try:
                interface.SetMasterVolumeLevelScalar(level / 100.0, None)
            except Exception as e:
                print(f"[VolumeController] apply error: {e}")
        return level


# ---------------------------------------------------------------------------
# Розміри масивів ключових точок
# ---------------------------------------------------------------------------
KEYPOINT_SIZE      = 126   # тільки координати (USE_DELTA=False)
KEYPOINT_FULL_SIZE = 252   # координати + дельта руху (USE_DELTA=True)

# Глобальний стан для дельта-обчислення.
# УВАГА: не thread-safe — тільки для single-threaded inference/collection.
_prev_keypoints: np.ndarray | None = None


def extract_keypoints(results, use_delta: bool | None = None) -> np.ndarray:
    """
    Витягує ключові точки обох рук з результатів MediaPipe.

    use_delta=None  → береться з config.py (USE_DELTA)
    use_delta=False → 126 значень: [lh(63), rh(63)]
    use_delta=True  → 252 значення: [lh(63), rh(63), delta_lh(63), delta_rh(63)]

    Координати нормовані відносно зап'ястя (landmark[0]).
    """
    global _prev_keypoints

    if use_delta is None:
        from src.config import USE_DELTA
        use_delta = USE_DELTA

    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            wrist = hand_landmarks.landmark[0]
            coords = np.array([
                [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                for lm in hand_landmarks.landmark
            ]).flatten()
            if label == 'Left':
                lh = coords
            else:
                rh = coords

    keypoints = np.concatenate([lh, rh])  # 126 float64

    if not use_delta:
        return keypoints

    # Дельта = поточний кадр − попередній (нуль якщо перший кадр)
    delta = np.zeros(126) if _prev_keypoints is None else keypoints - _prev_keypoints
    _prev_keypoints = keypoints.copy()
    return np.concatenate([keypoints, delta])  # 252


def reset_delta_state() -> None:
    """
    Скидає стан дельта-обчислення.
    Викликати при:
      - перемиканні режиму (GESTURES ↔ MOUSE)
      - паузі / зупинці захоплення
      - початку нової sequence під час збору даних
    """
    global _prev_keypoints
    _prev_keypoints = None


def check_pinch(hand_landmarks) -> bool:
    """
    Перевіряє щіпок: великий палець (4) + середній палець (12).
    Поріг: відстань < 0.03 у нормованих координатах.
    """
    thumb  = hand_landmarks.landmark[4]
    middle = hand_landmarks.landmark[12]
    return ((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2) ** 0.5 < 0.03