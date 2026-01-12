import cv2
import torch
import numpy as np
import pyautogui
import sys
import os
import mediapipe as mp

# Додаємо шлях до src, щоб працювали імпорти
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GestureLSTM
from src.utils import extract_keypoints, calculate_volume_level

# --- НАЛАШТУВАННЯ ---
gestures = ['static', 'swipe_right']
sequence_length = 30
threshold = 0.95  # Високий поріг для стабільності
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- ЗАВАНТАЖЕННЯ МОДЕЛІ ---
model = GestureLSTM(num_classes=len(gestures))
model.load_state_dict(torch.load('models/test_model.pth', map_location=device))
model.eval()

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def test_interface():
    cap = cv2.VideoCapture(0)
    
    sequence = []
    predictions = []
    cooldown = 0  
    last_vol = -1

    print("Система готова! Використовуй 'swipe_right' для музики або 'pinch' для гучності.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # 1. Отримуємо координати для LSTM
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        # 2. Логіка розпізнавання жестів (LSTM)
        if len(sequence) == sequence_length:
            input_data = torch.tensor(np.array([sequence]), dtype=torch.float32).to(device)
            with torch.no_grad():
                res = model(input_data)
                prob = torch.softmax(res, dim=1)
                max_prob, action_idx = torch.max(prob, 1)

                predictions.append(action_idx.item())
                predictions = predictions[-10:] 

                if max_prob.item() > threshold:
                    if len(predictions) >= 5 and all(p == action_idx.item() for p in predictions[-5:]):
                        detected_gesture = gestures[action_idx]

                        if detected_gesture == 'swipe_right' and cooldown == 0:
                            pyautogui.press('nexttrack')
                            print(f">>> ЖЕСТ: {detected_gesture.upper()} (Впевненість: {max_prob.item():.2f})")
                            cooldown = 30 #

        # 3. Логіка гучності (Математика)
        #if results.multi_hand_landmarks:
        #    for hand_landmarks in results.multi_hand_landmarks:
        #        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #        
        #        current_vol = calculate_volume_level(hand_landmarks)
        #        
        #        cv2.putText(frame, f"VOL: {current_vol}%", (50, 150), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


        if cooldown > 0:
            cooldown -= 1

        cv2.putText(frame, f"STATUS: READY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Hand Control System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_interface()