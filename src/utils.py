import numpy as np
import math

def get_distance(p1, p2):
    """Рахує евклідову відстань між двома точками в 3D (або 2D)"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_volume_level(hand_landmarks):
    """Вираховує рівень гучності на основі відстані між великим та вказівним пальцями"""
    # Точка 4 - Thumb tip, Точка 8 - Index finger tip
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    distance = get_distance(thumb_tip, index_tip)
    
    # Нормалізація: 0.03 (майже торкаються) до 0.25 (максимально розведені)
    # Перетворюємо у відсотки 0-100
    vol = np.interp(distance, [0.03, 0.25], [0, 100])
    return int(vol)

def extract_keypoints(results):
    """Перетворює дані MediaPipe від двох рук у єдиний масив для нейронки (126 значень)"""
    lh = np.zeros(21 * 3) # Ліва рука (21 точка * x,y,z)
    rh = np.zeros(21 * 3) # Права рука
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Визначаємо тип руки (Left/Right)
            label = results.multi_handedness[i].classification[0].label
            
            # Витягуємо координати точок
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            if label == 'Left':
                lh = coords
            else:
                rh = coords
                
    # Об'єднуємо: перші 63 значення - ліва, наступні 63 - права
    return np.concatenate([lh, rh])