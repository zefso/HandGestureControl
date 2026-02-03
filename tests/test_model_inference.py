import cv2
import torch
import numpy as np
import pyautogui
import sys
import os
import webbrowser
import mediapipe as mp
from collections import deque

# Додаємо шлях до src для імпортів
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import GestureLSTM
from src.utils import extract_keypoints

# --- CONFIGURATION ---
# Вместо ручного списка:
# gestures = ['static', 'swipe_right', 'swipe_left', 'ok', 'stop', 'browser']

# Используй тот же метод, что и в обучении:
DATA_PATH = 'data'
GESTURES = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
print(f"Loaded gestures in alphabetical order: {GESTURES}")
SEQ_LENGTH = 30
THRESHOLD = 0.92  # Вищий поріг для професійної точності
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join('models', 'gesture_lstm_best.pth')

# Вимикаємо затримки pyautogui для миттєвої реакції
pyautogui.PAUSE = 0

def check_pinch(hand_landmarks):
    """Геометрична перевірка з'єднання пальців"""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < 0.04

def test_interface():
    cap = cv2.VideoCapture(0)
    
    # Завантаження моделі
    model = GestureLSTM(num_classes=len(GESTURES)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded. Recognized gestures: {GESTURES}")
    else:
        print(f"Error: Model {MODEL_PATH} not found!")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    sequence = []
    predictions = []
    last_gesture = "static"
    cooldown = 0 
    prev_vol = 50 

    print("System started. 'q' - exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        probabilities = None

        # 1. ОБРОБКА ЖЕСТІВ (LSTM)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQ_LENGTH:]

        if len(sequence) == SEQ_LENGTH:
            input_data = torch.tensor(np.array([sequence]), dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                res = model(input_data)
                prob = torch.softmax(res, dim=1)
                probabilities = prob[0].cpu().numpy()
                max_prob, action_idx = torch.max(prob, 1)
                predictions.append(action_idx.item())
                predictions = predictions[-10:] 

                if max_prob.item() > THRESHOLD and len(predictions) >= 5:
                    if all(p == action_idx.item() for p in predictions[-5:]):
                        detected = GESTURES[action_idx]
                        
                        if detected != last_gesture and cooldown == 0:
                            if detected == 'swipe_left':
                                #pyautogui.hotkey('alt', 'tab') # Previous Window
                                print(">>> Move: swipe_left")
                                cooldown = 40 
                            elif detected == 'swipe_right':
                                #pyautogui.press('right') # Right Arrow
                                print(">>> Move: swipe_right")
                                cooldown = 40
                            elif detected == 'ok':
                                # Подвійна перевірка (LSTM + Geometry)
                                is_pinched = False
                                if results.multi_hand_landmarks:
                                    for hl in results.multi_hand_landmarks:
                                        if check_pinch(hl): is_pinched = True
                                
                                if is_pinched:
                                    print(">>> Action: ok")
                                    pyautogui.press('enter')
                                    cooldown = 60
                            elif detected == 'stop':
                                #pyautogui.press('esc')
                                print(">>> Action: stop")
                                cooldown = 40
                            elif detected == 'browser':
                                #webbrowser.open('https://www.google.com')
                                print(">>> Action: browser")
                                cooldown = 50
                        
                            last_gesture = detected

        # 2. VIZUALIZATION
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if cooldown > 0: cooldown -= 1
        
        # UI Overlay
        # 1. Background for text
        cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
        
        # 2. Status Text
        cv2.putText(frame, f"GESTURE: {last_gesture.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 3. Confidence Bar
        if probabilities is not None:
             for i, prob in enumerate(probabilities):
                cv2.rectangle(frame, (0, 60+i*40), (int(prob*100), 90+i*40), (245, 117, 16), -1)
                cv2.putText(frame, f'{GESTURES[i]}: {prob:.2f}', (0, 85+i*40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Hand Gesture Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_interface()