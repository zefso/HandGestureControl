import cv2
import torch
import numpy as np
import pyautogui
import sys
import os
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import webbrowser

# Додаємо шлях до src для імпортів
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GestureLSTM
from src.utils import extract_keypoints, calculate_volume_level

# --- НАЛАШТУВАННЯ ---
gestures = ['static', 'swipe_right', 'swipe_left', 'ok', 'stop']
sequence_length = 30

threshold = 0.92  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Ініціалізація Аудіо (Windows Core Audio API) ---
#def init_audio():
#    try:
#        devices = AudioUtilities.GetDevice(None)
#        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#        volume = cast(interface, POINTER(IAudioEndpointVolume))
#        print("Аудіо-система підключена.")
#        return volume
#    except Exception as e:
#        print(f"Помилка аудіо (перевірте пристрої відтворення): {e}")
#        return None

#volume_control = init_audio()

def test_interface():
    cap = cv2.VideoCapture(0)
    
    # Завантаження моделі
    model = GestureLSTM(num_classes=len(gestures)).to(device)
    model_path = 'models/test_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Модель {len(gestures)} жестів готова.")
    else:
        print("Файл моделі не знайдено!")
        return

    # MediaPipe налаштування
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    sequence = []
    predictions = []
    last_gesture = "static"
    cooldown = 0 

    print("Запуск відеопотоку... Натисніть 'q' для виходу.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # 1. Опрацювання жестів через нейронну мережу
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            input_data = torch.tensor(np.array([sequence]), dtype=torch.float32).to(device)
            with torch.no_grad():
                res = model(input_data)
                prob = torch.softmax(res, dim=1)
                max_prob, action_idx = torch.max(prob, 1)

                predictions.append(action_idx.item())
                predictions = predictions[-10:] 

                if max_prob.item() > threshold and len(predictions) >= 5:
                    if all(p == action_idx.item() for p in predictions[-5:]):
                        detected = gestures[action_idx]
                        
                        if detected != last_gesture and cooldown == 0:
                            if detected == 'swipe_right':
                                pyautogui.press('nexttrack')
                                cooldown = 35 
                            elif detected == 'swipe_left':
                                pyautogui.press('prevtrack')
                                cooldown = 35
                            elif detected == 'ok':
                                pyautogui.press('enter')
                                print(">>> ДІЯ: ENTER")
                                cooldown = 40
                            elif detected == 'stop':
                                pyautogui.press('playpause')
                                print(">>> ДІЯ: PLAY/PAUSE")
                                cooldown = 40
                            # ... у циклі розпізнавання жестів ...
                            elif detected == 'gesture_v':
                                # Відкриваємо браузер
                                webbrowser.open('https://www.google.com')
                                print(">>> ДІЯ: ВІДКРИТТЯ БРАУЗЕРА")
                                cooldown = 100 # Більший кулдаун, щоб не відкрити 20 вкладок

                            if detected != 'static':
                                print(f"Gesture: {detected.upper()} ({max_prob.item():.2f})")
                        
                        last_gesture = detected

        # 2. Математична модель для гучності (Pinch)
        #if results.multi_hand_landmarks:
        #    for hand_landmarks in results.multi_hand_landmarks:
        #        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #        
        #        # Обчислюємо гучність тільки якщо об'єкт volume_control ініціалізовано
        #        if volume_control:
        #            vol_percent = calculate_volume_level(hand_landmarks)
        #            volume_control.SetMasterVolumeLevelScalar(vol_percent / 100.0, None)
        #            cv2.putText(frame, f"VOLUME: {vol_percent}%", (50, 150), 
        #                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if cooldown > 0: cooldown -= 1
        
        cv2.putText(frame, f"LAST: {last_gesture.upper()}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Hand Gesture Control System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_interface()