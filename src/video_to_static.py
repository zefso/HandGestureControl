"""
video_to_static.py — інструмент для масового збору даних (наприклад, для 'static') 
з готового відеофайлу.

Використання:
  python src/video_to_static.py --video path/to/video.mp4 --target static
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import extract_keypoints, reset_delta_state
from src.config import GESTURES, SEQ_LENGTH, DATA_PATH, USE_DELTA

MAX_LOST_RATIO = 0.10  # Відкидаємо sequence, якщо >10% кадрів без руки


def first_empty_sequence(gesture: str) -> int:
    i = 0
    while True:
        if not os.path.exists(os.path.join(DATA_PATH, gesture, str(i), '0.npy')):
            return i
        i += 1


def process_video(video_path: str, target_gesture: str):
    if not os.path.exists(video_path):
        print(f"\n[Video2Static] ❌ Помилка: відео '{video_path}' не знайдено.")
        return

    if target_gesture not in GESTURES:
        print(f"\n[Video2Static] ⚠️ Увага: '{target_gesture}' немає в config.py.")
        ans = input("Все одно продовжити? (y/n): ").strip().lower()
        if ans != 'y':
            return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        print("[Video2Static] ❌ Неможливо прочитати відео.")
        return

    os.makedirs(os.path.join(DATA_PATH, target_gesture), exist_ok=True)
    seq_idx = first_empty_sequence(target_gesture)
    
    print("=" * 60)
    print("  VIDEO TO DATASET EXTRACTOR")
    print(f"  Відео: {video_path} ({total_frames} кадрів, ~{total_frames/fps:.1f} сек)")
    print(f"  Жест:  {target_gesture}")
    print(f"  Старт з індексу: {seq_idx}")
    print("=" * 60)

    mp_hands = mp.solutions.hands
    frames_buffer = []
    lost_count = 0
    saved_count = 0
    processed_frames = 0
    
    reset_delta_state()

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            
            # Для відео з вебкамери зазвичай треба flip, але якщо ви знімали
            # на телефон як звичайне відео - flip може перетворити праву руку на ліву.
            # Оскільки ми збираємо 'static' (шум), це не так критично, але 
            # залишаємо стандартний підхід:
            frame = cv2.flip(frame, 1)

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if not results.multi_hand_landmarks:
                lost_count += 1
                
            keypoints = extract_keypoints(results, use_delta=USE_DELTA)
            frames_buffer.append(keypoints)
            
            if len(frames_buffer) == SEQ_LENGTH:
                lost_ratio = lost_count / SEQ_LENGTH
                if lost_ratio <= MAX_LOST_RATIO:
                    save_path = os.path.join(DATA_PATH, target_gesture, str(seq_idx))
                    os.makedirs(save_path, exist_ok=True)
                    for i, kp in enumerate(frames_buffer):
                        np.save(os.path.join(save_path, f"{i}.npy"), kp)
                    
                    saved_count += 1
                    seq_idx += 1
                
                # Очищаємо буфер для наступної незалежної sequence
                frames_buffer = []
                lost_count = 0
                reset_delta_state()

            # Прогрес в консолі
            if processed_frames % 100 == 0:
                percent = (processed_frames / total_frames) * 100
                print(f"  Обробка: {percent:.1f}% | Збережено нових sequences: {saved_count}", end="\r")

    cap.release()
    print(f"\n\n[Video2Static] ✓ Готово! З відео витягнуто {saved_count} нових sequences.")
    print(f"Тепер у вас {seq_idx} прикладів для '{target_gesture}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Витягує навчальні sequences з відео")
    parser.add_argument('--video', type=str, required=True, 
                        help='media/test_video.mp4')
    parser.add_argument('--target', type=str, default='static', 
                        help='Жест, для якого збираємо дані (за замовчуванням static)')
    args = parser.parse_args()
    
    process_video(args.video, args.target)
