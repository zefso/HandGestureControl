"""
merge_datasets.py — об'єднання двох датасетів (свого і чужого).
Автоматично перенумеровує чужі файли щоб не перезаписати свої.

Запуск: python src/tools/merge_datasets.py --source "шлях/до/data/друга"
"""

import os
import shutil
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import DATA_PATH, GESTURES


def merge_datasets(source_dir: str):
    if not os.path.exists(source_dir):
        print(f"[Merge] ERROR: folder '{source_dir}' not found.")
        return

    print("=" * 60)
    print("  MERGE DATASETS")
    print(f"  Source: {os.path.abspath(source_dir)}")
    print(f"  Target: {os.path.abspath(DATA_PATH)}")
    print("=" * 60)

    total_copied = 0

    for gesture in GESTURES:
        src_gesture_path = os.path.join(source_dir, gesture)
        if not os.path.exists(src_gesture_path):
            continue

        tgt_gesture_path = os.path.join(DATA_PATH, gesture)
        os.makedirs(tgt_gesture_path, exist_ok=True)

        existing = [
            int(d) for d in os.listdir(tgt_gesture_path)
            if d.isdigit() and os.path.exists(os.path.join(tgt_gesture_path, d, '0.npy'))
        ]
        next_idx = max(existing) + 1 if existing else 0

        friend_seqs = sorted([
            int(d) for d in os.listdir(src_gesture_path)
            if d.isdigit() and os.path.exists(os.path.join(src_gesture_path, d, '0.npy'))
        ])
        if not friend_seqs:
            continue

        print(f"  {gesture:<15} | source: {len(friend_seqs)} seqs | appending from index {next_idx}")
        copied = 0

        for src_seq in friend_seqs:
            src_path = os.path.join(src_gesture_path, str(src_seq))
            tgt_path = os.path.join(tgt_gesture_path, str(next_idx))
            try:
                shutil.copytree(src_path, tgt_path)
                next_idx += 1
                copied += 1
                total_copied += 1
            except Exception as e:
                print(f"    ERROR copying {src_path}: {e}")

        print(f"    [OK] {copied} sequences added. Total now: {next_idx}")

    print("\n" + "=" * 60)
    print(f"[Merge] Done! {total_copied} new sequences added.")
    print("Now retrain: python src/train.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True,
                        help="Path to the other person's data/ folder")
    args = parser.parse_args()
    merge_datasets(args.source)
