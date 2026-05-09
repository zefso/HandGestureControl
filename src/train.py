"""
train.py — навчання GestureLSTM.

Можливості:
  - Перевірка розміру датасету перед стартом
  - Аугментація під час тренування (flip по X, шум)
  - LR Scheduler (ReduceLROnPlateau)
  - Early Stopping (patience=PATIENCE)
  - Збереження history у JSON
  - Матриця помилок + графіки
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import GestureLSTM
from src.config import (
    GESTURES, DATA_PATH, INPUT_SIZE, USE_DELTA,
    MODEL_DIR, DIAGRAM_DIR, MODEL_NAME,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, SEQ_LENGTH, PATIENCE,
    DEVICE,
)

# ---------------------------------------------------------------------------
# Аугментація
# ---------------------------------------------------------------------------

def augment_noise(window: np.ndarray, scale: float = 0.002) -> np.ndarray:
    """Додає невеликий гаусовий шум — підвищує стійкість до тремтіння."""
    return window + np.random.normal(0, scale, window.shape).astype(window.dtype)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GestureDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray,
                 gesture_names: list[str] | None = None,
                 augment: bool = False):
        self.data         = data
        self.labels       = labels
        self.gesture_names = gesture_names or GESTURES
        self.augment      = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        window = self.data[idx].copy()
        label  = self.labels[idx]

        if self.augment:
            # Невеликий шум для всіх жестів
            if np.random.rand() < 0.5:
                window = augment_noise(window)

        return torch.FloatTensor(window), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Завантаження даних
# ---------------------------------------------------------------------------

def validate_dataset_size() -> None:
    """
    Перевіряє що розмір збережених .npy відповідає INPUT_SIZE.
    Якщо ні — зупиняє виконання з чіткою помилкою.
    """
    for gesture in GESTURES:
        sample = os.path.join(DATA_PATH, gesture, '0', '0.npy')
        if os.path.exists(sample):
            actual = np.load(sample).shape[0]
            if actual != INPUT_SIZE:
                raise ValueError(
                    f"\n[train] ❌ Невідповідність розміру датасету!\n"
                    f"  Жест '{gesture}': знайдено {actual} ознак, очікується {INPUT_SIZE}\n"
                    f"  USE_DELTA={USE_DELTA} → INPUT_SIZE={INPUT_SIZE}\n"
                    f"\n  Рішення:\n"
                    f"    python src/smart_collector.py --clean   # очистити\n"
                    f"    python src/smart_collector.py           # зібрати знову\n"
                )


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(GESTURES)}

    for gesture in GESTURES:
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_path):
            print(f"[train] ⚠️  Папка не знайдена: {gesture_path}")
            continue

        sequences = sorted(os.listdir(gesture_path), key=lambda x: int(x) if x.isdigit() else x)
        print(f"  → '{gesture}': {len(sequences)} sequences")

        for seq in sequences:
            seq_path = os.path.join(gesture_path, seq)
            if not os.path.isdir(seq_path):
                continue

            frames = sorted(
                [f for f in os.listdir(seq_path) if f.endswith('.npy')],
                key=lambda x: int(x.split('.')[0])
            )

            window = []
            for frame in frames[:SEQ_LENGTH]:
                res = np.load(os.path.join(seq_path, frame))
                window.append(res)

            # Доповнюємо нулями якщо кадрів менше SEQ_LENGTH
            while len(window) < SEQ_LENGTH:
                window.append(np.zeros(INPUT_SIZE))

            X.append(window)
            y.append(label_map[gesture])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Тренування
# ---------------------------------------------------------------------------

def train_model() -> None:
    print(f"\n{'='*55}")
    print(f"  GESTURE LSTM TRAINER")
    print(f"  USE_DELTA={USE_DELTA} | INPUT_SIZE={INPUT_SIZE} | DEVICE={DEVICE}")
    print(f"  Жести ({len(GESTURES)}): {GESTURES}")
    print(f"{'='*55}\n")

    # 0. Валідація датасету
    validate_dataset_size()

    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(DIAGRAM_DIR, exist_ok=True)

    # 1. Завантаження
    print("Завантаження даних...")
    X, y = load_data()
    print(f"\n  Всього зразків: {len(X)}")
    if len(X) == 0:
        print("[train] ❌ Датасет порожній! Спочатку зберіть дані.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    train_loader = DataLoader(
        GestureDataset(X_train, y_train, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,  drop_last=False
    )
    test_loader = DataLoader(
        GestureDataset(X_test, y_test, augment=False),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # 2. Модель
    model = GestureLSTM(num_classes=len(GESTURES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-6
    )

    history = {'train_loss': [], 'test_acc': [], 'lr': []}
    best_acc    = 0.0
    no_improve  = 0
    best_epoch  = 0

    # 3. Цикл навчання
    print(f"\nНавчання ({EPOCHS} макс. епох, early stopping після {PATIENCE})...\n")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Валідація
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                _, preds = torch.max(model(inputs), 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        current_acc = accuracy_score(all_labels, all_preds)
        avg_loss    = running_loss / len(train_loader)
        cur_lr      = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_loss)
        history['test_acc'].append(current_acc)
        history['lr'].append(cur_lr)

        scheduler.step(current_acc)

        if (epoch + 1) % 10 == 0 or current_acc > best_acc:
            print(f"  Epoch {epoch+1:>4}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Acc: {current_acc*100:.2f}% | LR: {cur_lr:.2e}")

        # Збереження найкращої моделі
        if current_acc > best_acc:
            best_acc   = current_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_NAME))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping на епосі {epoch+1} (немає покращення {PATIENCE} епох)")
                break

    print(f"\n  ✓ Найкраща точність: {best_acc*100:.2f}% (epoch {best_epoch})")
    print(f"  Модель збережена: {os.path.join(MODEL_DIR, MODEL_NAME)}")

    # 4. Збереження history
    history_path = os.path.join(DIAGRAM_DIR, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'use_delta': USE_DELTA,
            'input_size': INPUT_SIZE,
            'gestures': GESTURES,
            'train_loss': history['train_loss'],
            'test_acc': history['test_acc'],
        }, f, indent=2)

    # 5. Графіки
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Loss', color='steelblue')
    plt.title('Training Loss')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot([a * 100 for a in history['test_acc']], label='Accuracy %', color='orange')
    plt.axhline(best_acc * 100, color='green', linestyle='--', label=f'Best {best_acc*100:.1f}%')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='LR', color='red')
    plt.title('Learning Rate')
    plt.xlabel('Epoch'); plt.yscale('log'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'learning_process.png'), dpi=120)
    plt.close()

    # Матриця помилок
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_NAME), map_location=DEVICE))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            _, preds = torch.max(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=GESTURES, yticklabels=GESTURES, cmap='Blues')
    plt.title('Confusion Matrix (best model)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'final_confusion_matrix.png'), dpi=120)
    plt.close()

    print(f"\n  Графіки збережені в: {DIAGRAM_DIR}")


if __name__ == "__main__":
    train_model()