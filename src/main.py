import cv2
import torch
import numpy as np
import pyautogui
import sys
import os
import webbrowser
import mediapipe as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model import GestureLSTM
    from src.utils import extract_keypoints, check_pinch, VolumeController
    from src.config import GESTURES, SEQ_LENGTH, MODEL_PATH, THRESHOLD, DEVICE, SWITCH_FRAMES
    from src.mouse_controller import AirMouse
except ImportError as e:
    print(f"Error  import: {e}. Check folder structure.")
    sys.exit()

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_system():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    mouse = AirMouse(smoothing=0.3)
    volume_ctrl = VolumeController()

    model = GestureLSTM(num_classes=len(GESTURES)).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found: {MODEL_PATH}")
        sys.exit()

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("System Ready. Modes: GESTURES <-> MOUSE (Hold 'ok'/'stop')")
    except Exception as e:
        print(f"Error: Model not found: {e}")
        sys.exit()

    return cap, mouse, volume_ctrl, model


# ---------------------------------------------------------------------------
# Hand processing
# ---------------------------------------------------------------------------

def process_hands(results):
    """Returns right hand landmarks and left fist flag."""
    right_hand_lms = None
    left_fist = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label
            lms = results.multi_hand_landmarks[idx]

            if label == 'Left':
                closed = all(lms.landmark[i].y > lms.landmark[i - 2].y for i in [8, 12, 16, 20])
                if closed and lms.landmark[0].y < 0.7:
                    left_fist = True
            else:
                right_hand_lms = lms

    return right_hand_lms, left_fist


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_gesture(model, sequence):
    input_data = torch.tensor(np.array([sequence]), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(input_data), dim=1)
        max_prob, action_idx = torch.max(probs, 1)
        if max_prob.item() > THRESHOLD:
            return GESTURES[action_idx.item()], max_prob.item()
    return "static", 0.0


# ---------------------------------------------------------------------------
# Mode switching
# ---------------------------------------------------------------------------

def handle_mode_switching(current_mode, detected, switch_counter):
    trigger = "ok" if current_mode == "GESTURES" else "stop"
    target = "MOUSE" if current_mode == "GESTURES" else "GESTURES"

    if detected == trigger:
        switch_counter += 1
        if switch_counter > SWITCH_FRAMES:
            print(f">>> SWITCHED TO {target} MODE")
            return target, 0, True, True   
        return current_mode, switch_counter, True, False
    else:
        switch_counter = max(0, switch_counter - 1)
        return current_mode, switch_counter, False, False


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def draw_interface(frame, mode, last_action, confidence, switching, switch_counter):
    h, w, _ = frame.shape
    color_map = {"VOLUME": (0, 255, 0), "MOUSE": (255, 0, 255)}
    UI_COLOR = color_map.get(mode, (245, 117, 16))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), UI_COLOR, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, f"MODE: {mode} | ACTION: {last_action.upper()}",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if switching:
        bar_w = int((switch_counter / SWITCH_FRAMES) * 150)
        cx, cy = w // 2, h - 50
        cv2.rectangle(frame, (cx - 75, cy), (cx - 75 + bar_w, cy + 10), (0, 255, 255), -1)
        cv2.rectangle(frame, (cx - 75, cy), (cx + 75, cy + 10), (255, 255, 255), 2)
        cv2.putText(frame, "HOLD TO SWITCH", (cx - 60, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if mode == "GESTURES" and confidence > 0 and not switching:
        bar_len = int(confidence * 150)
        color = (0, 0, 255) if confidence < 0.8 else (0, 255, 0)
        cv2.rectangle(frame, (w - 170, h - 30), (w - 20, h - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 170, h - 30), (w - 170 + bar_len, h - 10), color, -1)
        cv2.putText(frame, f"{int(confidence * 100)}%", (w - 210, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap, mouse, volume_ctrl, model = init_system()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    current_mode = "GESTURES"
    sequence = []
    last_gesture = "static"
    cooldown = 0
    fist_frames = 0
    switch_counter = 0

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            right_hand, left_fist = process_hands(results)
            detected = "static"
            confidence = 0.0

            # --- VOLUME MODE (left fist priority) ---
            if left_fist:
                fist_frames += 1
                if fist_frames >= 7:
                    if right_hand:
                        level = volume_ctrl.apply(right_hand)
                        last_gesture = f"VOL {level}%"
                    draw_interface(frame, "VOLUME", last_gesture, 0, False, 0)
                    if results.multi_hand_landmarks:
                        for hlms in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)
                    cv2.imshow('Gesture Control Hub', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
            else:
                fist_frames = 0

            # --- GESTURE RECOGNITION ---
            if results.multi_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQ_LENGTH:]

                if len(sequence) == SEQ_LENGTH and cooldown == 0:
                    detected, confidence = predict_gesture(model, sequence)

            # --- MODE SWITCHING ---
            current_mode, switch_counter, switching, reset = handle_mode_switching(
                current_mode, detected, switch_counter
            )
            if reset:
                sequence = []

            # --- ACTIONS BY MODE ---
            elif current_mode == "MOUSE":
                if right_hand:
                    try:
                        h_px, w_px, _ = frame.shape
                        mouse.move(right_hand)
                        state = mouse.handle_actions(right_hand)

                        cx = int(right_hand.landmark[8].x * w_px)
                        cy = int(right_hand.landmark[8].y * h_px)

                        indicators = {
                            "L_DOWN":      ((0, 255, 0),   20, True),
                            "RIGHT_CLICK": ((0, 0, 255),   25, True),
                        }
                        if state in indicators:
                            color, r, filled = indicators[state]
                            cv2.circle(frame, (cx, cy), r, color, -1 if filled else 2)
                        elif state == "SCROLLING":
                            cv2.arrowedLine(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 0), 2)
                        else:
                            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), 2)

                    except Exception as e:
                        print(f"Mouse Error: {e}")
                last_gesture = "MOUSE ACTIVE"

            elif current_mode == "GESTURES" and not switching:
                if detected not in ["static", "ok", "stop"] and cooldown == 0:
                    action_map = {
                        'swipe_left':  lambda: print(">>> SWIPE LEFT"),   # pyautogui.hotkey('alt', 'tab')
                        'swipe_right': lambda: print(">>> SWIPE RIGHT"),  # pyautogui.press('right')
                        'browser':     lambda: print(">>> BROWSER"),      # webbrowser.open('https://google.com')
                        'fist_left':   lambda: print(">>> FIST LEFT"),    # drag / grab
                    }
                    if detected in action_map:
                        action_map[detected]()
                    last_gesture = detected
                    cooldown = 40

            if cooldown > 0:
                cooldown -= 1

            # --- RENDER ---
            draw_interface(frame, current_mode, last_gesture, confidence, switching, switch_counter)

            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Gesture Control Hub', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()