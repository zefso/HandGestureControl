"""
merge_datasets.py — скрипт для об'єднання двох датасетів (наприклад, вашого і вашого друга).
Він автоматично перенумерує файли друга так, щоб вони не перезаписали ваші, а додалися в кінець.

Використання:
  python src/merge_datasets.py --source "шлях/до/папки/data/друга"
"""

import os
import shutil
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_PATH, GESTURES

def merge_datasets(source_dir: str):
    if not os.path.exists(source_dir):
        print(f"[Merge] ❌ Помилка: папку '{source_dir}' не знайдено.")
        return

    print("=" * 60)
    print("  MERGE DATASETS (ОБ'ЄДНАННЯ ДАНИХ)")
    print(f"  Звідки (папка друга): {os.path.abspath(source_dir)}")
    print(f"  Куди (ваша папка):    {os.path.abspath(DATA_PATH)}")
    print("=" * 60)

    total_copied = 0

    for gesture in GESTURES:
        src_gesture_path = os.path.join(source_dir, gesture)
        
        # Якщо у друга немає такого жесту — пропускаємо
        if not os.path.exists(src_gesture_path):
            continue

        tgt_gesture_path = os.path.join(DATA_PATH, gesture)
        os.makedirs(tgt_gesture_path, exist_ok=True)

        # Знаходимо, який індекс у вас наступний вільний
        existing_target_seqs = [
            int(d) for d in os.listdir(tgt_gesture_path) 
            if d.isdigit() and os.path.exists(os.path.join(tgt_gesture_path, d, '0.npy'))
        ]
        next_target_idx = max(existing_target_seqs) + 1 if existing_target_seqs else 0

        # Отримуємо всі записи друга
        friend_seqs = sorted([
            int(d) for d in os.listdir(src_gesture_path) 
            if d.isdigit() and os.path.exists(os.path.join(src_gesture_path, d, '0.npy'))
        ])
        
        if not friend_seqs:
            continue
            
        print(f"Жест '{gesture:<15}' | У друга: {len(friend_seqs)} записів | Ваші записи продовжуться з індексу {next_target_idx}...")
        copied_for_gesture = 0
        
        # Копіюємо і перенумеровуємо
        for src_seq in friend_seqs:
            src_path = os.path.join(src_gesture_path, str(src_seq))
            tgt_path = os.path.join(tgt_gesture_path, str(next_target_idx))
            
            try:
                shutil.copytree(src_path, tgt_path)
                next_target_idx += 1
                copied_for_gesture += 1
                total_copied += 1
            except Exception as e:
                print(f"  ❌ Помилка при копіюванні {src_path}: {e}")
                
        print(f"  ✓ Додано {copied_for_gesture} записів. Разом тепер {next_target_idx}.")

    print("\n" + "=" * 60)
    print(f"[Merge] Об'єднання завершено! Успішно додано {total_copied} нових sequences.")
    print("Тепер можете запускати тренування: python src/train.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Об'єднує ваш датасет з даними іншої людини")
    parser.add_argument('--source', type=str, required=True, 
                        help="Шлях до папки 'data' (або папки з жестами) вашого друга")
    args = parser.parse_args()
    
    merge_datasets(args.source)
