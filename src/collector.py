import cv2
import mediapipe as mp
import numpy as np
import os
import time
from utils import extract_keypoints

# --- НАЛАШТУВАННЯ ДАТАСЕТУ ---
DATA_PATH = os.path.join('data') 
# Твій фінальний набір жестів
gestures = np.array([
    'static',  
    #'swipe_left', 
    #'swipe_right',     
])

no_sequences = 60
sequence_length = 30 

# --- ІНІЦІАЛІЗАЦІЯ MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def collect_data():
    cap = cv2.VideoCapture(0)
    # Налаштовуємо на 2 руки
    with mp_hands.Hands(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5,
        max_num_hands=2
    ) as hands:
        
        for gesture in gestures:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    
                    # Візуалізація для зручності
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Логіка пауз перед початком кожного нового запису
                    if frame_num == 0:
                        cv2.putText(frame, 'GET READY!', (200, 200),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(frame, f'Collecting: {gesture.upper()}', (15, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, f'Video № {sequence}', (15, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collector', frame)
                        cv2.waitKey(1500) 
                    else:
                        cv2.putText(frame, f'RECORDING: {gesture}', (15, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collector', frame)

                    # Вилучення координат через нашу функцію (126 значень)
                    keypoints = extract_keypoints(results)
                    
                    target_dir = os.path.join(DATA_PATH, gesture, str(sequence))
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    
                    np.save(os.path.join(target_dir, str(frame_num)), keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    print("Збір даних завершено успішно!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()