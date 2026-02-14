import cv2
import torch
import numpy as np
import pyautogui
import sys
import os
import webbrowser
import mediapipe as mp

# Додаємо шлях до src для імпортів
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model import GestureLSTM
    from src.utils import extract_keypoints, calculate_volume_level, check_pinch, get_volume_interface
    from src.config import GESTURES, SEQ_LENGTH, MODEL_PATH, THRESHOLD, DEVICE, SWITCH_FRAMES
    from src.mouse_controller import AirMouse
except ImportError as e:
    print(f"Помилка імпорту: {e}. Перевірте структуру папок.")
    sys.exit()

volume = get_volume_interface()
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# --- MODULES ---
def init_system():
    """Ініціалізація ресурсів системи"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити камеру.")
        sys.exit()
    
    mouse = None
    try:
        mouse = AirMouse(smoothing=0.3)
    except Exception as e:
        print(f"Mouse Init Error: {e}")

    model = GestureLSTM(num_classes=len(GESTURES)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print(f"System Ready. Modes: GESTURES <-> MOUSE (Hold 'ok'/'stop')")
        except Exception as e:
            print(f"Model Load Error: {e}")
            sys.exit()
    else:
        print(f"Model not found: {MODEL_PATH}")
        sys.exit()

    return cap, mouse, model

def process_hands(results):
    """Аналіз результатів MediaPipe"""
    right_hand_lms = None
    left_fist = False
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label 
            lms = results.multi_hand_landmarks[idx]
            
            if label == 'Left':
                # Перевірка на кулак (всі пальці зігнуті)
                closed = all([lms.landmark[i].y > lms.landmark[i-2].y for i in [8, 12, 16, 20]])
                if closed and lms.landmark[0].y < 0.7: 
                    left_fist = True
            elif label == 'Right': 
                right_hand_lms = lms
                
    return right_hand_lms, left_fist

def predict_gesture(model, sequence):
    """Інференс LSTM моделі"""
    input_data = torch.tensor(np.array([sequence]), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        res = model(input_data)
        probs = torch.softmax(res, dim=1)
        max_prob, action_idx = torch.max(probs, 1)
        
        if max_prob.item() > THRESHOLD:
            return GESTURES[action_idx.item()], max_prob.item()
    return "static", 0.0

def handle_mode_switching(current_mode, detected, switch_counter, SWITCH_FRAMES):
    """Логіка перемикання між режимами"""
    switching = False
    new_mode = current_mode
    reset_sequence = False
    
    target_mode = "MOUSE" if current_mode == "GESTURES" else "GESTURES"
    trigger_gesture = "ok" if current_mode == "GESTURES" else "stop"

    if detected == trigger_gesture:
        switch_counter += 1
        switching = True
        if switch_counter > SWITCH_FRAMES:
            new_mode = target_mode
            switch_counter = 0
            reset_sequence = True
            print(f">>> SWITCHED TO {new_mode} MODE")
    else:
        if switch_counter > 0: switch_counter -= 1
        
    return new_mode, switch_counter, switching, reset_sequence

def draw_interface(frame, mode, last_action, confidence, switching, switch_counter, SWITCH_FRAMES):
    """Візуалізація UI"""
    h, w, _ = frame.shape
    
    # Колір теми залежно від режиму
    if mode == "VOLUME": UI_COLOR = (0, 255, 0)     
    elif mode == "MOUSE": UI_COLOR = (255, 0, 255)   
    else: UI_COLOR = (245, 117, 16)                 

    # Верхня панель
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), UI_COLOR, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    status_text = f"MODE: {mode} | ACTION: {last_action.upper()}"
    cv2.putText(frame, status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Індикатор перемикання
    if switching:
        bar_width = int((switch_counter / SWITCH_FRAMES) * 150)
        cx, cy = w // 2, h - 50
        cv2.rectangle(frame, (cx - 75, cy), (cx - 75 + bar_width, cy + 10), (0, 255, 255), -1)
        cv2.rectangle(frame, (cx - 75, cy), (cx + 75, cy + 10), (255, 255, 255), 2)
        cv2.putText(frame, "HOLD TO SWITCH", (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Індикатор впевненості (тільки для жестів)
    if mode == "GESTURES" and confidence > 0 and not switching:
        bar_len = int(confidence * 150)
        conf_color = (0, 0, 255) if confidence < 0.8 else (0, 255, 0)
        cv2.rectangle(frame, (w - 170, h - 30), (w - 20, h - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 170, h - 30), (w - 170 + bar_len, h - 10), conf_color, -1)
        cv2.putText(frame, f"{int(confidence*100)}%", (w - 210, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    cap, mouse, model = init_system()
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Стан
    current_mode = "GESTURES"
    sequence = []
    last_gesture = "static"
    cooldown = 0
    fist_frames = 0
    switch_counter = 0

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # 1. Обробка
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            right_hand, left_fist = process_hands(results)
            detected = "static"
            confidence = 0.0

            # 2. Логіка режимів
            # АУДІО (Пріоритет)
            if left_fist:
                fist_frames += 1
                if fist_frames >= 7:
                    current_mode_display = "VOLUME" # For display only
                    if right_hand:
                        try:
                            levels = calculate_volume_level(right_hand)
                            if volume: volume.SetMasterVolumeLevelScalar(levels / 100.0, None)
                            last_gesture = f"VOL {levels}%"
                        except Exception as e:
                            print(f"Volume Error: {e}")
                    draw_interface(frame, current_mode_display, last_gesture, 0, False, 0, SWITCH_FRAMES)
            else:
                fist_frames = 0
                
                # РОЗПІЗНАВАННЯ
                # розпізнаємо завжди для перемикання, або якщо в режимі жестів
                if results.multi_hand_landmarks:
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-SEQ_LENGTH:]
                    
                    if len(sequence) == SEQ_LENGTH and cooldown == 0:
                        detected, confidence = predict_gesture(model, sequence)
                
                # ПЕРЕМИКАННЯ
                current_mode, switch_counter, switching, reset = handle_mode_switching(current_mode, detected, switch_counter, SWITCH_FRAMES)
                if reset: sequence = []

                # ДІЇ
                elif current_mode == "MOUSE":
                    if right_hand and mouse:
                        try:
                            h, w, _ = frame.shape 
                        
                            mouse.move(right_hand)
                            mouse_state = mouse.handle_actions(right_hand)
                            
                            cx, cy = int(right_hand.landmark[8].x * w), int(right_hand.landmark[8].y * h)
                        
                            if mouse_state == "L_DOWN":
                                cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1) # Зелений - тягнемо
                            elif mouse_state == "RIGHT_CLICK":
                                cv2.circle(frame, (cx, cy), 25, (0, 0, 255), -1) # Червоний - правый клік
                            elif mouse_state == "SCROLLING":
                                cv2.arrowedLine(frame, (cx, cy-20), (cx, cy+20), (255, 255, 0), 2) # Стрілки - скрол
                            else:
                                cv2.circle(frame, (cx, cy), 10, (255, 0, 255), 2) # Фіолетовий - спокій
                            
                        except Exception as e:
                            print(f"Mouse Error: {e}")
                    last_gesture = "MOUSE ACTIVE"

                elif current_mode == "GESTURES" and not switching:
                    if detected not in ["static", "ok", "stop"] and cooldown == 0:
                        if detected == 'swipe_left': 
                            print(">>> SWIPE LEFT")
                            # pyautogui.hotkey('alt', 'tab')
                        elif detected == 'swipe_right': 
                            print(">>> SWIPE RIGHT")
                            # pyautogui.press('right')
                        elif detected == 'browser': 
                            print(">>> BROWSER")
                            # webbrowser.open('https://google.com')
                        
                        last_gesture = detected
                        cooldown = 40
                
                if cooldown > 0: cooldown -= 1
                
                # ВІЗУАЛІЗАЦІЯ
                draw_interface(frame, current_mode, last_gesture, confidence, switching, switch_counter, SWITCH_FRAMES)
            
            # Малюємо скелет рук
            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)
            
            cv2.imshow('Gesture Control Hub', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()