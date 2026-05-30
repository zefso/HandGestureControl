"""
smart_collector.py — збір навчальних даних з підказками варіативності і
                     перевіркою якості та сумісності датасету.

Керування:
  SPACE — записати sequence
  N     — наступний жест
  P     — попередній жест
  D     — видалити останню sequence
  Q     — вийти

CLI:
  python src/smart_collector.py           — звичайний запуск
  python src/smart_collector.py --clean   — очистити ВСЕ дані і почати знову
  python src/smart_collector.py --target static --count 200 — зібрати 200 прикладів для одного жесту
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import shutil
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import extract_keypoints, reset_delta_state
from src.config import GESTURES, SEQ_LENGTH, DATA_PATH, USE_DELTA, INPUT_SIZE

# ---------------------------------------------------------------------------
# Налаштування
# ---------------------------------------------------------------------------

MAX_LOST_RATIO = 0.10       # максимум 10% кадрів без руки

# Жести де очікується ОДНА рука (права). Дві руки — попередження.
SINGLE_HAND_GESTURES = {'swipe_left', 'swipe_right', 'ok', 'stop', 'browser'}

# Підказки варіативності — циклічно перед кожною серією
VARIABILITY_HINTS = [
    "Normal distance from camera",
    "Get CLOSER to the camera",
    "Move FARTHER from the camera",
    "Hand slightly HIGHER than usual",
    "Hand slightly LOWER than usual",
    "Tilt hand slightly left",
    "Tilt hand slightly right",
    "Move SLOWLY",
    "Move QUICKLY",
    "Change lighting angle (tilt head)",
]

# Кольори UI (BGR)
CLR_OK     = (0, 210, 80)
CLR_WARN   = (0, 180, 255)
CLR_ERR    = (0, 60, 220)
CLR_BG     = (20, 20, 20)
CLR_WHITE  = (255, 255, 255)
CLR_YELLOW = (0, 220, 220)
CLR_CYAN   = (200, 220, 0)

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ---------------------------------------------------------------------------
# Перевірка сумісності датасету
# ---------------------------------------------------------------------------

def check_dataset_consistency() -> bool:
    """
    Перевіряє розмір існуючих .npy файлів.
    Якщо розмір не відповідає INPUT_SIZE — попереджає та повертає False.
    Повертає True якщо все гаразд або датасет порожній.
    """
    mismatch_found = False
    for gesture in GESTURES:
        sample_path = os.path.join(DATA_PATH, gesture, '0', '0.npy')
        if os.path.exists(sample_path):
            actual = np.load(sample_path).shape[0]
            if actual != INPUT_SIZE:
                print(
                    f"[Collector] ⚠️  MISMATCH DATASET: '{gesture}' has size {actual}, "
                    f"expected {INPUT_SIZE} (USE_DELTA={USE_DELTA})"
                )
                mismatch_found = True

    if mismatch_found:
        print()
        print("  Old dataset is NOT COMPATIBLE with current USE_DELTA!")
        print("  Options:")
        print("    1. Run with --clean to clear and collect again")
        print("    2. Change USE_DELTA in config.py back to the previous value")
        print()
        return False
    return True

# ---------------------------------------------------------------------------
# Очищення датасету
# ---------------------------------------------------------------------------

def clean_dataset() -> None:
    """Видаляє всю папку data/ і створює порожню структуру."""
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        print(f"[Collector] Датасет очищено: {DATA_PATH}")


def create_folders(target_gesture: str | None, count: int) -> None:
    gestures_to_create = [target_gesture] if target_gesture else GESTURES
    for gesture in gestures_to_create:
        for seq in range(count):
            os.makedirs(os.path.join(DATA_PATH, gesture, str(seq)), exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def first_empty_sequence(gesture: str, max_count: int) -> int:
    for i in range(max_count):
        if not os.path.exists(os.path.join(DATA_PATH, gesture, str(i), '0.npy')):
            return i
    return max_count


def count_recorded(gesture: str, max_count: int) -> int:
    count = 0
    for i in range(max_count):
        if os.path.exists(os.path.join(DATA_PATH, gesture, str(i), '0.npy')):
            count += 1
    return count


def delete_last_sequence(gesture: str, max_count: int) -> bool:
    for i in range(max_count - 1, -1, -1):
        path = os.path.join(DATA_PATH, gesture, str(i))
        if os.path.exists(os.path.join(path, '0.npy')):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            print(f"[Collector] Видалено sequence {i} для '{gesture}'")
            return True
    return False


# ---------------------------------------------------------------------------
# UI функції
# ---------------------------------------------------------------------------

def draw_top_bar(frame, gesture: str, seq_idx: int, recorded: int, max_count: int) -> None:
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 65), CLR_BG, -1)
    cv2.putText(frame, f"GESTURE: {gesture.upper()}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_WHITE, 2)
    pct = recorded / max_count if max_count > 0 else 0
    bar_x, bar_y, bar_w, bar_h = 12, 40, w - 110, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * pct), bar_y + bar_h), CLR_OK, -1)
    cv2.putText(frame, f"{recorded}/{max_count}",
                (bar_x + bar_w + 8, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1)


def draw_hint(frame, hint: str, seq_idx: int) -> None:
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h - 80), (w, h - 50), (30, 30, 60), -1)
    cv2.putText(frame, f"HINT #{(seq_idx % len(VARIABILITY_HINTS)) + 1}: {hint}",
                (10, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_YELLOW, 1)


def draw_controls(frame) -> None:
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h - 48), (w, h), CLR_BG, -1)
    controls = "SPACE=Record  N=Next  P=Prev  D=Delete last  Q=Quit"
    cv2.putText(frame, controls, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def draw_two_hands_warning(frame) -> None:
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 70), (w, 105), (0, 0, 180), -1)
    cv2.putText(frame, "! REMOVE SECOND HAND — only ONE hand required here!",
                (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.58, CLR_WHITE, 2)


def _show_rejection_screen(cap, message: str) -> None:
    deadline = time.time() + 1.0  # показувати 1 секунду
    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, message, (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 2)
        cv2.imshow('Smart Collector', frame)
        cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Record one sequence
# ---------------------------------------------------------------------------

def record_sequence(cap, hands, gesture: str, seq_idx: int, max_count: int) -> bool:
    """
    Записує SEQ_LENGTH кадрів.
    Повертає True якщо sequence якісна (lost < MAX_LOST_RATIO).
    """
    reset_delta_state()

    frames_data: list[np.ndarray] = []
    lost_frames = 0
    hint = VARIABILITY_HINTS[seq_idx % len(VARIABILITY_HINTS)]
    is_single_hand = gesture in SINGLE_HAND_GESTURES

    # --- Плавний зворотній відлік 3..1 ---
    for countdown in range(2, 0, -1):
        deadline = time.time() + 0.7
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            draw_top_bar(frame, gesture, seq_idx, count_recorded(gesture, max_count), max_count)
            cv2.putText(frame, str(countdown), (w // 2 - 25, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, CLR_YELLOW, 6)
            cv2.imshow('Smart Collector', frame)
            cv2.waitKey(1)

    # --- Запис ---
    frame_count = 0
    while frame_count < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Попередження про дві руки під час запису
        if (results.multi_hand_landmarks
                and len(results.multi_hand_landmarks) > 1
                and is_single_hand):
            draw_two_hands_warning(frame)

        # Скелет руки
        if results.multi_hand_landmarks:
            for hlms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

        # Якість: рахуємо кадри без руки
        if not results.multi_hand_landmarks:
            lost_frames += 1

        keypoints = extract_keypoints(results, use_delta=USE_DELTA)
        frames_data.append(keypoints)
        frame_count += 1

        # Відображення UI
        draw_top_bar(frame, gesture, seq_idx, count_recorded(gesture, max_count), max_count)
        draw_hint(frame, hint, seq_idx)

        progress = frame_count / SEQ_LENGTH
        h, w, _ = frame.shape
        rec_y = 110
        cv2.rectangle(frame, (0, rec_y), (w, rec_y + 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, rec_y), (int(w * progress), rec_y + 10), CLR_ERR, -1)
        cv2.putText(frame, f"REC {frame_count}/{SEQ_LENGTH}",
                    (10, rec_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_ERR, 2)

        lost_pct = lost_frames / frame_count
        q_color = CLR_OK if lost_pct < MAX_LOST_RATIO else CLR_WARN
        cv2.putText(frame, f"LOST: {int(lost_pct * 100)}%",
                    (w - 130, rec_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, q_color, 2)

        draw_controls(frame)
        cv2.imshow('Smart Collector', frame)
        cv2.waitKey(1)

    # --- Перевірка якості ---
    lost_ratio = lost_frames / SEQ_LENGTH
    if lost_ratio > MAX_LOST_RATIO:
        msg = f"REJECTED — {int(lost_ratio * 100)}% frames without hand (max {int(MAX_LOST_RATIO * 100)}%)"
        print(f"[Collector] ❌ Sequence {seq_idx} | {msg}")
        _show_rejection_screen(cap, msg)
        return False

    # --- Збереження ---
    save_path = os.path.join(DATA_PATH, gesture, str(seq_idx))
    os.makedirs(save_path, exist_ok=True)
    for i, kp in enumerate(frames_data):
        np.save(os.path.join(save_path, f"{i}.npy"), kp)

    print(f"[Collector] ✓ {gesture} seq={seq_idx} | lost={int(lost_ratio * 100)}% | size={INPUT_SIZE}")
    return True


# ---------------------------------------------------------------------------
# Головний цикл
# ---------------------------------------------------------------------------

def collect_data(target_gesture: str | None, max_count: int) -> None:
    create_folders(target_gesture, max_count)

    if not check_dataset_consistency():
        ans = input("Продовжити все одно? (y/n): ").strip().lower()
        if ans != 'y':
            return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[Collector] Помилка: камера не знайдена.")
        return

    gestures_list = [target_gesture] if target_gesture else GESTURES

    print("=" * 55)
    print("  SMART DATA COLLECTOR")
    print(f"  USE_DELTA = {USE_DELTA} | INPUT_SIZE = {INPUT_SIZE}")
    print(f"  Кількість sequences на жест: {max_count}")
    print(f"  Жести: {gestures_list}")
    print("=" * 55)
    print("  SPACE — record sequence")
    print("  N     — next gesture")
    print("  P     — previous gesture")
    print("  D     — delete last sequence")
    print("  Q     — Quit")
    print("=" * 55)

    gesture_idx = 0

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            gesture  = gestures_list[gesture_idx]
            recorded = count_recorded(gesture, max_count)
            seq_idx  = first_empty_sequence(gesture, max_count)

            # Попередження про дві руки в режимі очікування
            if (results.multi_hand_landmarks
                    and len(results.multi_hand_landmarks) > 1
                    and gesture in SINGLE_HAND_GESTURES):
                draw_two_hands_warning(frame)

            # Скелет
            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

            draw_top_bar(frame, gesture, seq_idx, recorded, max_count)

            h, w, _ = frame.shape
            if seq_idx < max_count:
                hint = VARIABILITY_HINTS[seq_idx % len(VARIABILITY_HINTS)]
                draw_hint(frame, hint, seq_idx)
                cv2.putText(frame, "Press SPACE to record", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_OK, 2)
            else:
                cv2.putText(frame, "DONE! Press N for next gesture", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_YELLOW, 2)

            draw_controls(frame)
            cv2.imshow('Smart Collector', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n'):
                gesture_idx = (gesture_idx + 1) % len(gestures_list)
                reset_delta_state()
                print(f"[Collector] → '{gestures_list[gesture_idx]}'")
            elif key == ord('p'):
                gesture_idx = (gesture_idx - 1) % len(gestures_list)
                reset_delta_state()
                print(f"[Collector] → '{gestures_list[gesture_idx]}'")
            elif key == ord('d'):
                if not delete_last_sequence(gesture, max_count):
                    print("[Collector] Нема що видаляти.")
            elif key == 32:  # SPACE
                if seq_idx >= max_count:
                    print(f"[Collector] '{gesture}' вже має {max_count} sequences.")
                    continue
                success = record_sequence(cap, hands, gesture, seq_idx, max_count)
                if not success:
                    print("[Collector] Sequence відкинута, спробуй ще раз.")
                reset_delta_state()

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Collector] Збір завершено!")
    _print_summary(gestures_list, max_count)


def _print_summary(gestures_list: list, max_count: int) -> None:
    print("\n--- ПІДСУМОК ---")
    total = 0
    for gesture in gestures_list:
        recorded = count_recorded(gesture, max_count)
        total   += recorded
        bar   = "█" * int((recorded / max_count) * 20) if max_count > 0 else ""
        bar  += "░" * (20 - len(bar))
        status = "✓" if recorded >= max_count else f"⚠ {recorded}/{max_count}"
        print(f"  {gesture:<15} {status}  [{bar}]")
    print(f"\n  Всього sequences: {total} / {len(gestures_list) * max_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Data Collector")
    parser.add_argument('--clean', action='store_true',
                        help='Очистити весь датасет і почати знову')
    parser.add_argument('--target', type=str, default=None,
                        help='Назва конкретного жесту для збору (наприклад, static або swipe_right)')
    parser.add_argument('--count', type=int, default=200,
                        help='Скільки прикладів зібрати (за замовчуванням 200)')
    args = parser.parse_args()

    if args.target and args.target not in GESTURES:
        print(f"[Collector] Помилка: жест '{args.target}' не знайдений в config.py.")
        print(f"Доступні жести: {GESTURES}")
        sys.exit(1)

    if args.clean:
        confirm = input(f"Видалити весь датасет в '{DATA_PATH}'? (yes/no): ").strip()
        if confirm == 'yes':
            clean_dataset()
            print("[Collector] Датасет очищено. Починаємо збір.")
            create_folders(args.target, args.count)
        else:
            print("[Collector] Скасовано.")
            sys.exit(0)

    collect_data(args.target, args.count)
