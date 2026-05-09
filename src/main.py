import cv2
import torch
import numpy as np
import pyautogui
import sys
import os

# Додаємо корінь проекту до sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import mediapipe as mp
    from src.model import GestureLSTM
    from src.utils import extract_keypoints, VolumeController, reset_delta_state
    from src.hotkey_executor import GestureActionExecutor
    from src.config import (
        GESTURES, SEQ_LENGTH, MODEL_PATH, THRESHOLD, DEVICE,
        SWITCH_FRAMES, COOLDOWN_FRAMES, MOUSE_SMOOTHING,
    )
    from src.mouse_controller import AirMouse
except ImportError as e:
    print(f"[main] Помилка імпорту: {e}")
    sys.exit(1)

# Налаштування PyAutoGUI
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# ─────────────────────────────────────────────────────────────
# DRY_RUN = True  →  всі дії ТІЛЬКИ пишуться в консоль,
#                    нічого реального не виконується.
# Перемикається клавішею D під час роботи.
# ─────────────────────────────────────────────────────────────
DRY_RUN = True


def init_system():
    """Ініціалізація всіх компонентів системи."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[main] Помилка: не вдалося відкрити камеру.")
        sys.exit(1)

    mouse       = AirMouse()
    volume_ctrl = VolumeController()

    # Конфіг і виконавець жестів — gestures.json лежить поряд з main.py (в src/)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gestures.json')
    executor = GestureActionExecutor(config_path=config_path)
    print(executor.summary())

    model = GestureLSTM(num_classes=len(GESTURES)).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"[main] Модель не знайдена: {MODEL_PATH}")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"[main] System Ready | Gestures: {GESTURES}")
    except Exception as e:
        print(f"[main] Помилка завантаження моделі: {e}")
        sys.exit(1)

    return cap, mouse, volume_ctrl, executor, model


def process_hands(results):
    """Повертає (right_hand_lms, left_fist_bool)."""
    right_hand_lms = None
    left_fist = False
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label
            lms   = results.multi_hand_landmarks[idx]
            if label == 'Left':
                closed = all(lms.landmark[i].y > lms.landmark[i - 2].y for i in [8, 12, 16, 20])
                if closed and lms.landmark[0].y < 0.7:
                    left_fist = True
            else:
                right_hand_lms = lms
    return right_hand_lms, left_fist


def predict_gesture(model, sequence):
    """Запускає LSTM інференс. Повертає (gesture_name, confidence)."""
    input_data = torch.tensor(np.array([sequence]), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(input_data), dim=1)
        max_prob, action_idx = torch.max(probs, 1)
        if max_prob.item() > THRESHOLD:
            return GESTURES[action_idx.item()], max_prob.item()
    return "static", 0.0


def handle_mode_switching(current_mode, detected, switch_counter):
    """
    Логіка перемикання GESTURES ↔ MOUSE.
    Потрібно утримувати жест 'ok' (з GESTURES) або 'stop' (з MOUSE)
    протягом SWITCH_FRAMES кадрів.

    Повертає: (new_mode, switch_counter, is_switching, did_switch)
    """
    trigger = "ok"   if current_mode == "GESTURES" else "stop"
    target  = "MOUSE" if current_mode == "GESTURES" else "GESTURES"

    if detected == trigger:
        switch_counter += 1
        if switch_counter > SWITCH_FRAMES:
            print(f">>> SWITCHED TO {target} MODE")
            return target, 0, True, True
        return current_mode, switch_counter, True, False

    switch_counter = max(0, switch_counter - 1)
    return current_mode, switch_counter, False, False


def draw_interface(frame, mode, last_action, confidence,
                   switching, switch_counter, active_profile, dry_run=False):
    h, w, _ = frame.shape
    color_map = {"VOLUME": (0, 200, 100), "MOUSE": (200, 0, 200)}
    ui_color  = color_map.get(mode, (200, 100, 20))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), ui_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"MODE: {mode} | {last_action.upper()}",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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

    # Бар впевненості моделі
    if mode == "GESTURES" and confidence > 0 and not switching:
        bar_len   = int(confidence * 150)
        bar_color = (0, 0, 220) if confidence < 0.8 else (0, 200, 0)
        cv2.rectangle(frame, (w - 170, h - 30), (w - 20, h - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 170, h - 30), (w - 170 + bar_len, h - 10), bar_color, -1)
        cv2.putText(frame, f"{int(confidence*100)}%", (w - 210, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # DRY_RUN банер
    if dry_run:
        cv2.rectangle(frame, (0, h - 22), (220, h), (0, 0, 0), -1)
        cv2.putText(frame, "[D] DRY RUN — no actions", (5, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


def main():
    global DRY_RUN
    cap, mouse, volume_ctrl, executor, model = init_system()
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    current_mode   = "GESTURES"
    sequence       = []
    last_gesture   = "READY"
    cooldown       = 0
    fist_frames    = 0
    switch_counter = 0

    if DRY_RUN:
        print("\n[DRY RUN] Режим спостереження — жодних реальних дій. Натисни D щоб вимкнути.\n")

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Клавіатурне керування
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                DRY_RUN = not DRY_RUN
                status = "УВІМКНЕНО" if DRY_RUN else "ВИМКНЕНО"
                print(f"[DRY RUN] {status}")
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

            results     = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            right_hand, left_fist = process_hands(results)
            detected    = "static"
            confidence  = 0.0

            # Логіка гучності (лівий кулак + права рука)
            if left_fist:
                fist_frames += 1
                if fist_frames >= 7 and right_hand:
                    # Рахуємо рівень завжди (щоб бачити в UI)
                    level = volume_ctrl.calculate_level(right_hand)
                    if DRY_RUN:
                        print(f"[DRY RUN] VOLUME → {level}%")
                    else:
                        volume_ctrl.apply(right_hand)
                    last_gesture = f"VOL {level}%"
                    draw_interface(frame, "VOLUME", last_gesture, 0, False, 0,
                                   executor.active_profile, DRY_RUN)
                    if results.multi_hand_landmarks:
                        for hlms in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)
                    cv2.imshow('Gesture Control Hub', frame)
                    continue
            else:
                fist_frames = 0

            # Збір ключових точок
            if results.multi_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQ_LENGTH:]
                if len(sequence) == SEQ_LENGTH and cooldown == 0:
                    detected, confidence = predict_gesture(model, sequence)

            # Перемикання режимів
            current_mode, switch_counter, switching, did_switch = handle_mode_switching(
                current_mode, detected, switch_counter
            )

            if did_switch:
                sequence = []
                reset_delta_state()   # ← скидаємо дельта-стан при перемиканні

            # Режим Миші
            elif current_mode == "MOUSE":
                if right_hand:
                    state = mouse.handle_actions(right_hand)  # розраховуємо завжди
                    if DRY_RUN:
                        if state != "IDLE":
                            print(f"[DRY RUN] MOUSE → {state}")
                    else:
                        mouse.move(right_hand)
                        # handle_actions вже викликаний вище, повторно не треба
                    h_px, w_px, _ = frame.shape
                    cx = int(right_hand.landmark[8].x * w_px)
                    cy = int(right_hand.landmark[8].y * h_px)
                    if state == "L_DOWN":
                        cv2.circle(frame, (cx, cy), 15, (0, 200, 0), -1)
                    elif state == "RIGHT_CLICK":
                        cv2.circle(frame, (cx, cy), 20, (0, 0, 220), -1)
                    elif state == "SCROLLING":
                        cv2.arrowedLine(frame, (cx, cy - 20), (cx, cy + 20), (200, 200, 0), 2)
                    else:
                        cv2.circle(frame, (cx, cy), 10, (200, 0, 200), 2)
                last_gesture = "MOUSE ACTIVE"

            # Режим Жестів
            elif current_mode == "GESTURES" and not switching:
                if detected not in ["static", "ok", "stop"] and cooldown == 0:
                    if DRY_RUN:
                        # Беремо опис дії з конфігу, але не виконуємо
                        actions = executor._profiles.get(executor.active_profile, {}).get("actions", {})
                        action_cfg = actions.get(detected, {})
                        desc = action_cfg.get("description", detected)
                        print(f"[DRY RUN] GESTURE → {detected}  ({desc})")
                        last_gesture = f"{detected}: {desc}"
                    else:
                        description  = executor.execute(detected)
                        last_gesture = description
                    cooldown = COOLDOWN_FRAMES

            if cooldown > 0:
                cooldown -= 1

            # Малювання UI і скелету
            draw_interface(frame, current_mode, last_gesture, confidence,
                           switching, switch_counter, executor.active_profile, DRY_RUN)
            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Gesture Control Hub', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()