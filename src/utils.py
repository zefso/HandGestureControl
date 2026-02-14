import numpy as np
import math

# Distance calculation
def get_distance(p1, p2):
    """Рахує евклідову відстань між двома точками в 3D (або 2D)"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Volume control
_last_vol = 0

def calculate_volume_level(hand_landmarks, smoothing=0.15):
    global _last_vol
    if not hand_landmarks:
        return _last_vol
    
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    distance = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
    
    current_vol = np.interp(distance, [0.03, 0.26], [0, 100])
    
    if current_vol >= 97: 
        current_vol = 100  
        return 100
    if current_vol <= 3: 
        current_vol = 0
        return 0
    
    if abs(current_vol - _last_vol) < 1.5:
        return _last_vol
        
    smoothed_vol = int(smoothing * current_vol + (1 - smoothing) * _last_vol)
    _last_vol = smoothed_vol
    return smoothed_vol


# Keypoints extraction
def extract_keypoints(results):
    """Перетворює дані MediaPipe від двох рук у єдиний масив для нейронки (126 значень)"""
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3) 

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            label = results.multi_handedness[i].classification[0].label
            wrist = hand_landmarks.landmark[0] 
            res = []
            for lm in hand_landmarks.landmark:
                res.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            coords = np.array(res).flatten()

            
            if label == 'Left':
                lh = coords
            else:
                rh = coords
                
    return np.concatenate([lh, rh])


# Click detection 
def check_pinch(hand_landmarks):
    thumb = hand_landmarks.landmark[4]
    middle = hand_landmarks.landmark[12]
    
    distance = ((thumb.x - middle.x)**2 + (thumb.y - middle.y)**2)**0.5
    
    if distance < 0.03:
        return True
    return False


import comtypes
from comtypes import CLSCTX_ALL, GUID
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IMMDeviceEnumerator
from ctypes import cast, POINTER

# Volume control
def get_volume_interface():
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception:
        try:
            CLSID_MMDeviceEnumerator = GUID('{BCDE0395-E52F-467C-8E3D-C4579291692E}')
            IID_IMMDeviceEnumerator = GUID('{A95664D2-9614-4F35-A746-DE8DB63617E6}')
            enumerator = comtypes.CoCreateInstance(
                CLSID_MMDeviceEnumerator,
                IMMDeviceEnumerator,
                comtypes.CLSCTX_INPROC_SERVER
            )
            endpoint = enumerator.GetDefaultAudioEndpoint(0, 0)
            interface = endpoint.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            print(f"Критична помилка аудіо-інтерфейсу: {e}")
            return None
