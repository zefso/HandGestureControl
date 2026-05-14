"""
data_analysis.py — аналіз якості зібраного датасету.

Перевіряє:
  - Кількість sequences на жест
  - Фактичний розмір .npy файлів (чи відповідає INPUT_SIZE)
  - Відсоток порожніх кадрів (рука не виявлена)
  - Загальну статистику

Запуск: python src/tools/data_analysis.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import DATA_PATH, INPUT_SIZE, USE_DELTA, GESTURES, SEQ_LENGTH


def analyze_data() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"[DataAnalysis] Папка '{DATA_PATH}' не існує. Спочатку зберіть дані.")
        return

    print(f"\n{'='*60}")
    print(f"  DATA ANALYSIS")
    print(f"  USE_DELTA={USE_DELTA} | Очікуваний INPUT_SIZE={INPUT_SIZE}")
    print(f"  Шлях: {DATA_PATH}")
    print(f"{'='*60}\n")

    total_sequences  = 0
    total_frames     = 0
    total_lost       = 0
    size_mismatches  = 0

    for gesture in GESTURES:
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_path):
            print(f"  [!] '{gesture}': папка відсутня")
            continue

        sequences = sorted(
            [s for s in os.listdir(gesture_path)
             if os.path.isdir(os.path.join(gesture_path, s)) and s.isdigit()],
            key=int
        )
        seq_count    = len(sequences)
        lost_frames  = 0
        frames_total = 0
        wrong_size   = 0

        for seq in sequences:
            seq_path = os.path.join(gesture_path, seq)
            frames   = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
            frames_total += len(frames)

            for fname in frames:
                fpath = os.path.join(seq_path, fname)
                data  = np.load(fpath)

                if data.shape[0] != INPUT_SIZE:
                    wrong_size += 1
                    continue

                coord_part = data[:126]
                if np.all(coord_part == 0):
                    lost_frames += 1

        loss_rate = (lost_frames / frames_total * 100) if frames_total > 0 else 0.0
        total_sequences += seq_count
        total_frames    += frames_total
        total_lost      += lost_frames
        size_mismatches += wrong_size

        status = "[OK]" if loss_rate <= 10 and wrong_size == 0 else "[!!]"
        print(f"  {status} {gesture:<15} | seq: {seq_count:>3} | "
              f"frames: {frames_total:>5} | lost: {loss_rate:>5.1f}%", end="")
        if wrong_size > 0:
            print(f" | WRONG SIZE: {wrong_size} (expected {INPUT_SIZE})", end="")
        if loss_rate > 10:
            print(f" | HIGH LOSS", end="")
        print()

    global_loss = (total_lost / total_frames * 100) if total_frames > 0 else 0.0
    print(f"\n{'─'*60}")
    print(f"  Total sequences : {total_sequences}")
    print(f"  Total frames    : {total_frames}")
    print(f"  Global loss     : {global_loss:.2f}%")
    if size_mismatches > 0:
        print(f"\n  ERROR: {size_mismatches} files with wrong size!")
        print(f"  Run: python src/smart_collector.py --clean")
    else:
        print(f"\n  [OK] All files have correct size ({INPUT_SIZE})")


if __name__ == "__main__":
    analyze_data()
