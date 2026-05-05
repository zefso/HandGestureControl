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
# Settings
# ---------------------------------------------------------------------------

NO_SEQUENCES   = 100        # sequences per gesture
MAX_LOST_RATIO = 0.10       # max % of lost frames

# Gestures where ONE hand (right) is expected. Two hands - warning.
SINGLE_HAND_GESTURES = {'swipe_left', 'swipe_right', 'ok', 'stop', 'browser'}

# Variability hints - cyclically before each series
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

# UI colors (BGR)
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
# Dataset consistency check
# ---------------------------------------------------------------------------

def check_dataset_consistency() -> bool:
    """
    Checks the size of existing .npy files.
    If size does not match INPUT_SIZE - warns and returns False.
    Returns True if everything is ok or dataset is empty.
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
# Clean dataset
# ---------------------------------------------------------------------------

def clean_dataset() -> None:
    """Removes the entire data/ folder and creates an empty structure."""
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        print(f"[Collector] Dataset cleaned: {DATA_PATH}")
    create_folders()


def create_folders() -> None:
    for gesture in GESTURES:
        for seq in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, gesture, str(seq)), exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def first_empty_sequence(gesture: str) -> int:
    for i in range(NO_SEQUENCES):
        if not os.path.exists(os.path.join(DATA_PATH, gesture, str(i), '0.npy')):
            return i
    return NO_SEQUENCES


def count_recorded(gesture: str) -> int:
    count = 0
    for i in range(NO_SEQUENCES):
        if os.path.exists(os.path.join(DATA_PATH, gesture, str(i), '0.npy')):
            count += 1
    return count


def delete_last_sequence(gesture: str) -> bool:
    for i in range(NO_SEQUENCES - 1, -1, -1):
        path = os.path.join(DATA_PATH, gesture, str(i))
        if os.path.exists(os.path.join(path, '0.npy')):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            print(f"[Collector] Deleted sequence {i} for '{gesture}'")
            return True
    return False


# ---------------------------------------------------------------------------
# UI functions
# ---------------------------------------------------------------------------

def draw_top_bar(frame, gesture: str, seq_idx: int, recorded: int) -> None:
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 65), CLR_BG, -1)
    cv2.putText(frame, f"GESTURE: {gesture.upper()}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_WHITE, 2)
    pct = recorded / NO_SEQUENCES
    bar_x, bar_y, bar_w, bar_h = 12, 40, w - 24, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * pct), bar_y + bar_h), CLR_OK, -1)
    cv2.putText(frame, f"{recorded}/{NO_SEQUENCES}",
                (bar_x + bar_w + 6, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1)


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
    deadline = time.time() + 1.0  # show for 1 second
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

def record_sequence(cap, hands, gesture: str, seq_idx: int) -> bool:
    """
    Records SEQ_LENGTH frames.
    Returns True if sequence is high quality (lost < MAX_LOST_RATIO).
    """
    reset_delta_state()

    frames_data: list[np.ndarray] = []
    lost_frames = 0
    hint = VARIABILITY_HINTS[seq_idx % len(VARIABILITY_HINTS)]
    is_single_hand = gesture in SINGLE_HAND_GESTURES

    # --- Smooth countdown 3..1 ---
    for countdown in range(3, 0, -1):
        deadline = time.time() + 0.7
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            draw_top_bar(frame, gesture, seq_idx, count_recorded(gesture))
            cv2.putText(frame, str(countdown), (w // 2 - 25, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, CLR_YELLOW, 6)
            cv2.imshow('Smart Collector', frame)
            cv2.waitKey(1)

    # --- Recording ---
    frame_count = 0
    while frame_count < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Warning for two hands
        if (results.multi_hand_landmarks
                and len(results.multi_hand_landmarks) > 1
                and is_single_hand):
            draw_two_hands_warning(frame)

        # Skeleton
        if results.multi_hand_landmarks:
            for hlms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

        # Quality
        if not results.multi_hand_landmarks:
            lost_frames += 1

        keypoints = extract_keypoints(results, use_delta=USE_DELTA)
        frames_data.append(keypoints)
        frame_count += 1

        # UI
        draw_top_bar(frame, gesture, seq_idx, count_recorded(gesture))
        draw_hint(frame, hint, seq_idx)

        progress = frame_count / SEQ_LENGTH
        h, w, _ = frame.shape
        rec_y = 110
        cv2.rectangle(frame, (0, rec_y), (w, rec_y + 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, rec_y), (int(w * progress), rec_y + 10), CLR_ERR, -1)
        cv2.putText(frame, f"REC {frame_count}/{SEQ_LENGTH}",
                    (10, rec_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_ERR, 2)

        lost_pct = lost_frames / max(frame_count, 1)
        q_color = CLR_OK if lost_pct < MAX_LOST_RATIO else CLR_WARN
        cv2.putText(frame, f"LOST: {int(lost_pct * 100)}%",
                    (w - 130, rec_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, q_color, 2)

        draw_controls(frame)
        cv2.imshow('Smart Collector', frame)
        cv2.waitKey(1)

    # --- Quality check ---
    lost_ratio = lost_frames / max(SEQ_LENGTH, 1)
    if lost_ratio > MAX_LOST_RATIO:
        msg = f"REJECTED — {int(lost_ratio * 100)}% frames without hand (max {int(MAX_LOST_RATIO * 100)}%)"
        print(f"[Collector] ❌ Sequence {seq_idx} | {msg}")
        _show_rejection_screen(cap, msg)
        return False

    # --- Save ---
    save_path = os.path.join(DATA_PATH, gesture, str(seq_idx))
    os.makedirs(save_path, exist_ok=True)
    for i, kp in enumerate(frames_data):
        np.save(os.path.join(save_path, f"{i}.npy"), kp)

    print(f"[Collector] ✓ {gesture} seq={seq_idx} | lost={int(lost_ratio * 100)}% | size={INPUT_SIZE}")
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def collect_data() -> None:
    create_folders()

    if not check_dataset_consistency():
        ans = input("Continue anyway? (y/n): ").strip().lower()
        if ans != 'y':
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Collector] Error: camera not found.")
        return

    print("=" * 55)
    print("  SMART DATA COLLECTOR")
    print(f"  USE_DELTA = {USE_DELTA} | INPUT_SIZE = {INPUT_SIZE}")
    print(f"  Gestures: {GESTURES}")
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

            gesture  = GESTURES[gesture_idx]
            recorded = count_recorded(gesture)
            seq_idx  = first_empty_sequence(gesture)

            # Warning for two hands in idle mode
            if (results.multi_hand_landmarks
                    and len(results.multi_hand_landmarks) > 1
                    and gesture in SINGLE_HAND_GESTURES):
                draw_two_hands_warning(frame)

            # Skeleton
            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)

            draw_top_bar(frame, gesture, seq_idx, recorded)

            h, w, _ = frame.shape
            if seq_idx < NO_SEQUENCES:
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
                gesture_idx = (gesture_idx + 1) % len(GESTURES)
                reset_delta_state()
                print(f"[Collector] → '{GESTURES[gesture_idx]}'")
            elif key == ord('p'):
                gesture_idx = (gesture_idx - 1) % len(GESTURES)
                reset_delta_state()
                print(f"[Collector] → '{GESTURES[gesture_idx]}'")
            elif key == ord('d'):
                if not delete_last_sequence(gesture):
                    print("[Collector] Nothing to delete.")
            elif key == 32:  # SPACE
                if seq_idx >= NO_SEQUENCES:
                    print(f"[Collector] '{gesture}' already has {NO_SEQUENCES} sequences.")
                    continue
                success = record_sequence(cap, hands, gesture, seq_idx)
                if not success:
                    print("[Collector] Sequence rejected, try again.")
                reset_delta_state()

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Collector] Collection finished!")
    _print_summary()


def _print_summary() -> None:
    print("\n--- SUMMARY ---")
    total = 0
    for gesture in GESTURES:
        recorded = count_recorded(gesture)
        total   += recorded
        bar   = "█" * recorded + "░" * (NO_SEQUENCES - recorded)
        status = "✓" if recorded >= NO_SEQUENCES else f"⚠ {recorded}/{NO_SEQUENCES}"
        print(f"  {gesture:<15} {status}  [{bar[:20]}]")
    print(f"\n  Total sequences: {total} / {len(GESTURES) * NO_SEQUENCES}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Data Collector")
    parser.add_argument('--clean', action='store_true',
                        help='Delete entire dataset and start over')
    args = parser.parse_args()

    if args.clean:
        confirm = input(f"Delete entire dataset in '{DATA_PATH}'? (yes/no): ").strip()
        if confirm == 'yes':
            clean_dataset()
            print("[Collector] Dataset cleaned. Starting collection.")
        else:
            print("[Collector] Cancelled.")
            sys.exit(0)

    collect_data()
