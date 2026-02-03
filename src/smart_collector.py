import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# Добавляем путь к src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import extract_keypoints

# --- CONFIG ---
DATA_PATH = os.path.join('data')
gestures = ['static', 'swipe_right', 'swipe_left', 'ok', 'stop', 'browser']
sequence_length = 30
no_sequences = 100

# --- SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

def create_folders():
    for gesture in gestures:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, gesture, str(sequence)))
            except:
                pass

def collect_data():
    create_folders()
    cap = cv2.VideoCapture(0)
    
    print("=== SMART DATA COLLECTOR ===")
    print("Controls:")
    print("  'n' - Next gesture")
    print("  'r' - Record current sequence (hold until finished)")
    print("  'q' - Quit")
    
    current_gesture_idx = 0
    current_sequence = 0
    
    # Find first empty sequence for current gesture
    while os.path.exists(os.path.join(DATA_PATH, gestures[current_gesture_idx], str(current_sequence), '0.npy')):
        current_sequence += 1
        
    recording = False
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Info Overlay
        cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
        
        gesture_name = gestures[current_gesture_idx]
        cv2.putText(frame, f"TARGET: {gesture_name.upper()} | SEQ: {current_sequence}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not recording:
             cv2.putText(frame, "Press 'SPACE' to Start Recording", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
             
             # Navigation
             key = cv2.waitKey(1) & 0xFF
             if key == ord('n'):
                 current_gesture_idx = (current_gesture_idx + 1) % len(gestures)
                 current_sequence = 0
                 # Find next empty
                 while os.path.exists(os.path.join(DATA_PATH, gestures[current_gesture_idx], str(current_sequence), '0.npy')):
                    current_sequence += 1
             elif key == ord('q'):
                 break
             elif key == 32: # SPACE
                 recording = True
                 frame_count = 0
                 
        if recording:
            if not results.multi_hand_landmarks:
                # Якщо рука зникла - скидаємо запис цієї послідовності
                # Це гарантує, що 30 кадрів будуть йти один за одним без "дирок"
                frame_count = 0
                cv2.putText(frame, "HAND LOST! RESETTING SEQUENCE...", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                keypoints = extract_keypoints(results)
                # Перевірка чи ми не вилізли за межі створених папок
                save_path = os.path.join(DATA_PATH, gesture_name, str(current_sequence))
                if not os.path.exists(save_path): os.makedirs(save_path)
                
                np.save(os.path.join(save_path, f"{frame_count}.npy"), keypoints)
                frame_count += 1
                cv2.putText(frame, f"RECORDING: {frame_count}/{sequence_length}", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if frame_count == sequence_length:
                    recording = False
                    current_sequence += 1
                    print(f"Recorded sequence {current_sequence-1} for {gesture_name}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.imshow('Smart Collector', frame)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
