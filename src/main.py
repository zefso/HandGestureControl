import cv2
import torch
import numpy as np
import pyautogui
import sys
import os
import webbrowser
import mediapipe as mp

# Додаємо шлях до модулів
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model import GestureLSTM
    from src.utils import extract_keypoints, VolumeController
    from src.hotkey_executor import GestureActionExecutor
    from src.config import (
        GESTURES, GESTURE_ACTIONS,
        SEQ_LENGTH, MODEL_PATH, THRESHOLD, DEVICE,
        SWITCH_FRAMES, COOLDOWN_FRAMES,
        MOUSE_SMOOTHING,
    )
    from src.mouse_controller import AirMouse
except ImportError as e:
    print(f"Помилка імпорту: {e}")
    sys.exit()

# Налаштування PyAutoGUI
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

def init_system():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити камеру.")
        sys.exit()

    mouse = AirMouse(smoothing=MOUSE_SMOOTHING)
    volume_ctrl = VolumeController()
    
    # Ініціалізація екзекутора з підтримкою профілів
    config_path = os.path.join(os.path.dirname(__file__), '..', 'gestures.json')
    executor = GestureActionExecutor(config_path=config_path)
    print(executor.summary())

    model = GestureLSTM(num_classes=len(GESTURES)).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Модель не знайдена: {MODEL_PATH}")
        sys.exit()

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"System Ready | Gestures: {GESTURES}")
    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")
        sys.exit()

    return cap, mouse, volume_ctrl, executor, model


def process_hands(results):
    right_hand_lms = None
    left_fist = False
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label
            lms = results.multi_hand_landmarks[idx]
            if label == 'Left':
                # Перевірка на кулак
                closed = all(lms.landmark[i].y > lms.landmark[i - 2].y for i in [8, 12, 16, 20])
                if closed and lms.landmark[0].y < 0.7:
                    left_fist = True
            else:
                right_hand_lms = lms
    return right_hand_lms, left_fist


def predict_gesture(model, sequence):
    input_data = torch.tensor(np.array([sequence]), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(input_data), dim=1)
        max_prob, action_idx = torch.max(probs, 1)
        if max_prob.item() > THRESHOLD:
            return GESTURES[action_idx.item()], max_prob.item()
    return "static", 0.0


def handle_mode_switching(current_mode, detected, switch_counter):
    trigger = "ok" if current_mode == "GESTURES" else "stop"
    target  = "MOUSE" if current_mode == "GESTURES" else "GESTURES"
    
    if detected == trigger:
        switch_counter += 1
        if switch_counter > SWITCH_FRAMES:
            print(f">>> SWITCHED TO {target} MODE")
            return target, 0, True, True
        return current_mode, switch_counter, True, False
    
    switch_counter = max(0, switch_counter - 1)
    return current_mode, switch_counter, False, False


def draw_interface(frame, mode, last_action, confidence, switching, switch_counter, active_profile):
    h, w, _ = frame.shape
    color_map = {"VOLUME": (0, 200, 100), "MOUSE": (200, 0, 200)}
    UI_COLOR = color_map.get(mode, (200, 100, 20))
    
    # Верхня панель
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), UI_COLOR, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, f"MODE: {mode} | {last_action.upper()}",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Відображення активного профілю
    cv2.putText(frame, f"PROFILE: {active_profile.upper()}",
                (w - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2)

    # Прогрес-бар перемикання режимів
    if switching:
        bar_w = int((switch_counter / SWITCH_FRAMES) * 150)
        cx, cy = w // 2, h - 50
        cv2.rectangle(frame, (cx - 75, cy), (cx - 75 + bar_w, cy + 10), (0, 230, 230), -1)
        cv2.rectangle(frame, (cx - 75, cy), (cx + 75, cy + 10), (255, 255, 255), 2)
        cv2.putText(frame, "HOLD TO SWITCH", (cx - 60, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 230), 2)

    # Впевненість моделі
    if mode == "GESTURES" and confidence > 0 and not switching:
        bar_len = int(confidence * 150)
        color = (0, 0, 220) if confidence < 0.8 else (0, 200, 0)
        cv2.rectangle(frame, (w - 170, h - 30), (w - 20, h - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 170, h - 30), (w - 170 + bar_len, h - 10), color, -1)


def main():
    cap, mouse, volume_ctrl, executor, model = init_system()
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    current_mode = "GESTURES"
    sequence = []
    last_gesture = "READY"
    cooldown = 0
    fist_frames = 0
    switch_counter = 0

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Обробка клавіш керування профілями
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                executor.next_profile()
            elif key == ord('1'):
                executor.set_profile("default")
            elif key == ord('2'):
                executor.set_profile("browser")
            elif key == ord('3'):
                executor.set_profile("media")
            elif key == ord('4'):
                executor.set_profile("vscode")
            elif key == ord('r'):
                executor.reload()

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            right_hand, left_fist = process_hands(results)
            detected = "static"
            confidence = 0.0

            # Логіка керування гучністю (лівий кулак + права рука)
            if left_fist:
                fist_frames += 1
                if fist_frames >= 7 and right_hand:
                    level = volume_ctrl.apply(right_hand)
                    last_gesture = f"VOL {level}%"
                    draw_interface(frame, "VOLUME", last_gesture, 0, False, 0, executor.active_profile)
                    cv2.imshow('Gesture Control Hub', frame)
                    continue
            else:
                fist_frames = 0

            # Збір скелета для жестів
            if results.multi_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQ_LENGTH:]
                if len(sequence) == SEQ_LENGTH and cooldown == 0:
                    detected, confidence = predict_gesture(model, sequence)

            # Перемикання режимів (Gesture <-> Mouse)
            current_mode, switch_counter, switching, reset = handle_mode_switching(
                current_mode, detected, switch_counter)
            
            if reset:
                sequence = []

            # Режим Миші
            elif current_mode == "MOUSE":
                if right_hand:
                    mouse.move(right_hand)
                    state = mouse.handle_actions(right_hand)
                    # Візуалізація стану миші
                    h_px, w_px, _ = frame.shape
                    cx, cy = int(right_hand.landmark[8].x * w_px), int(right_hand.landmark[8].y * h_px)
                    cv2.circle(frame, (cx, cy), 10, (200, 0, 200), 2)
                last_gesture = "MOUSE ACTIVE"

            # Режим Жестів
            elif current_mode == "GESTURES" and not switching:
                if detected not in ["static", "ok", "stop"] and cooldown == 0:
                    description = executor.execute(detected)
                    last_gesture = description
                    cooldown = COOLDOWN_FRAMES

            if cooldown > 0: cooldown -= 1

            # Малювання UI
            draw_interface(frame, current_mode, last_gesture, confidence, switching, switch_counter, executor.active_profile)
            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Gesture Control Hub', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()