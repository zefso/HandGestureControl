import numpy as np
import math

def get_distance(p1, p2):
    """Рахує евклідову відстань між двома точками в 3D (або 2D)"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_volume_level(hand_landmarks):
    if not hand_landmarks:
        return 0
    
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    
    distance = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
    
    vol = np.interp(distance, [0.02, 0.15], [0, 100])
    return int(vol)

def extract_keypoints(results):
    """Перетворює дані MediaPipe від двох рук у єдиний масив для нейронки (126 значень)"""
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3) 

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            label = results.multi_handedness[i].classification[0].label
            wrist = hand_landmarks.landmark[0] # Точка зап'ястя
            res = []
            for lm in hand_landmarks.landmark:
                res.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            coords = np.array(res).flatten()

            
            if label == 'Left':
                lh = coords
            else:
                rh = coords
                
    return np.concatenate([lh, rh])


def check_click(hand_landmarks):
    thumb = hand_landmarks.landmark[4]
    middle = hand_landmarks.landmark[12]
    
    distance = ((thumb.x - middle.x)**2 + (thumb.y - middle.y)**2)**0.5
    
    if distance < 0.03:
        return True
    return False