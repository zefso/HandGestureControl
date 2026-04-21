import numpy as np


# ---------------------------------------------------------------------------
# Volume control
# ---------------------------------------------------------------------------

class VolumeController:
    def __init__(self, smoothing: float = 0.15):
        self.smoothing = smoothing
        self._last_vol: int = 0
        self._interface = None

    def get_interface(self):
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
                IID_IMMDeviceEnumerator = GUID('{A95664D2-9614-4F35-A746-DE8DB63617E6}')
                enumerator = comtypes.CoCreateInstance(
                    CLSID_MMDeviceEnumerator,
                    IMMDeviceEnumerator,
                    comtypes.CLSCTX_INPROC_SERVER,
                )
                endpoint = enumerator.GetDefaultAudioEndpoint(0, 0)
                interface = endpoint.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._interface = cast(interface, POINTER(IAudioEndpointVolume))

        except Exception as e:
            print(f"Critical error of audio interface: {e}")
            self._interface = None

        return self._interface

    def calculate_level(self, hand_landmarks) -> int:
        if not hand_landmarks:
            return self._last_vol

        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        distance = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5

        current_vol = float(np.interp(distance, [0.02, 0.20], [0, 100]))

        if current_vol >= 97:
            self._last_vol = 100
            return 100
        if current_vol <= 3:
            self._last_vol = 0
            return 0

        if abs(current_vol - self._last_vol) < 1.5:
            return self._last_vol

        smoothed = int(self.smoothing * current_vol + (1 - self.smoothing) * self._last_vol)
        self._last_vol = smoothed
        return smoothed

    def apply(self, hand_landmarks) -> int:
        level = self.calculate_level(hand_landmarks)
        interface = self.get_interface()
        if interface:
            try:
                interface.SetMasterVolumeLevelScalar(level / 100.0, None)
            except Exception as e:
                print(f"Volume apply error: {e}")
        return level


# ---------------------------------------------------------------------------
# Keypoints extraction
# ---------------------------------------------------------------------------

def extract_keypoints(results) -> np.ndarray:
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

    return np.concatenate([lh, rh])


# ---------------------------------------------------------------------------
# Click / pinch detection
# ---------------------------------------------------------------------------

def check_pinch(hand_landmarks) -> bool:
    thumb = hand_landmarks.landmark[4]
    middle = hand_landmarks.landmark[12]
    distance = ((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2) ** 0.5
    return distance < 0.03