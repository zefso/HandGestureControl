import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import GestureLSTM
from src.config import (
    GESTURES, DATA_PATH, INPUT_SIZE, MODEL_DIR, DIAGRAM_DIR,
    MODEL_NAME, BATCH_SIZE, LEARNING_RATE, EPOCHS, SEQ_LENGTH, USE_DELTA, PATIENCE
)

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(DIAGRAM_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Датасет
# ---------------------------------------------------------------------------

def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Аугментація одної послідовності (SEQ_LENGTH, INPUT_SIZE).
    Застосовується тільки під час тренування.
    """
    seq = seq.copy()

    # 1. Гаусівський шум — імітує тремтіння руки / неточність трекінгу
    if np.random.rand() < 0.5:
        seq += np.random.normal(0, 0.005, seq.shape).astype(np.float32)

    # 2. Масштабування — рука ближче/далі від камери
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.85, 1.15)
        seq *= scale

    # 3. Часовий зсув — починаємо жест трохи раніше або пізніше
    if np.random.rand() < 0.4:
        shift = np.random.randint(-3, 4)   # -3..+3 кадри
        seq = np.roll(seq, shift, axis=0)
        # Нулюємо кадри, що "провалились" крізь межу
        if shift > 0:
            seq[:shift] = 0
        elif shift < 0:
            seq[shift:] = 0

    return seq


class GestureDataset(Dataset):
    def __init__(self, data, labels, augment: bool = False):
        self.data    = data.astype(np.float32)
        self.labels  = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        if self.augment:
            seq = augment_sequence(seq)
        return torch.FloatTensor(seq), self.labels[idx]


# ---------------------------------------------------------------------------
# Завантаження даних
# ---------------------------------------------------------------------------

def load_data():
    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(GESTURES)}

    print(f"\nUSE_DELTA={USE_DELTA} | INPUT_SIZE={INPUT_SIZE}")
    print(f"Жести: {GESTURES}\n")

    for gesture in GESTURES:
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_path):
            print(f"  [!] Папка не знайдена: {gesture_path}")
            continue

        sequences = sorted(os.listdir(gesture_path))
        loaded = 0

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
                kp = np.load(os.path.join(seq_path, frame))

                # Захист від розмірності — якщо файл старого формату (126) а треба 252
                if kp.shape[0] != INPUT_SIZE:
                    if USE_DELTA and kp.shape[0] == 126:
                        # Доповнюємо нулями (дельта = 0)
                        kp = np.concatenate([kp, np.zeros(126)])
                    else:
                        kp = kp[:INPUT_SIZE]  # обрізаємо якщо більше

                window.append(kp)

            while len(window) < SEQ_LENGTH:
                window.append(np.zeros(INPUT_SIZE))

            X.append(window)
            y.append(label_map[gesture])
            loaded += 1

        print(f"  [{gesture:<14}] sequences завантажено: {loaded}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Class weights — вирівнювання дисбалансу
# ---------------------------------------------------------------------------

def compute_weights(y_train: np.ndarray):
    """
    Рахує два типи вагів:

    1. class_weights для CrossEntropyLoss — штрафує за помилки
       на рідких класах сильніше. static має 653 seq, swipe — 100,
       тому loss для swipe буде в ~6.5x більший.

    2. sample_weights для WeightedRandomSampler — під час кожної епохи
       samples з рідких класів вибираються частіше, щоб батчі були
       збалансованими.
    """
    classes = np.arange(len(GESTURES))

    # sklearn рахує weight = N / (n_classes * count_per_class)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.FloatTensor(cw)

    # Для кожного sample — вага його класу
    sample_weights = np.array([cw[label] for label in y_train])

    print("\n--- Class weights ---")
    for i, gesture in enumerate(GESTURES):
        count = np.sum(y_train == i)
        print(f"  {gesture:<15} count={count:>4}  weight={cw[i]:.3f}")
    print()

    return class_weights_tensor, sample_weights


# ---------------------------------------------------------------------------
# Тренування
# ---------------------------------------------------------------------------

def train_model():
    # 1. Дані
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Ваги
    class_weights, sample_weights = compute_weights(y_train)

    # WeightedRandomSampler — збалансовані батчі
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        GestureDataset(X_train, y_train, augment=True),
        batch_size=BATCH_SIZE,
        sampler=sampler       # замість shuffle=True
    )
    test_loader = DataLoader(
        GestureDataset(X_test, y_test, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 3. Модель
    from src.config import DEVICE
    model = GestureLSTM(num_classes=len(GESTURES)).to(DEVICE)

    # CrossEntropyLoss з class weights — штрафує за помилки на рідких класах
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # LR scheduler — зменшує learning rate якщо accuracy не росте
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=15, factor=0.5
    )

    history = {'train_loss': [], 'test_acc': [], 'per_class_acc': []}
    best_acc      = 0.0
    patience_left = PATIENCE   # early stopping

    # 4. Цикл навчання
    print(f"Починаємо навчання на {EPOCHS} епох (early stop patience={PATIENCE})...")
    print(f"Train: {len(X_train)} (з аугментацією) | Test: {len(X_test)}\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # стабільність LSTM
            optimizer.step()
            running_loss += loss.item()

        # Валідація
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        current_acc = accuracy_score(all_labels, all_preds)
        avg_loss    = running_loss / len(train_loader)

        history['train_loss'].append(avg_loss)
        history['test_acc'].append(current_acc)

        # Per-class accuracy — слідкуємо за свайпами окремо
        per_class = {}
        for i, gesture in enumerate(GESTURES):
            mask = np.array(all_labels) == i
            if mask.sum() > 0:
                per_class[gesture] = accuracy_score(
                    np.array(all_labels)[mask],
                    np.array(all_preds)[mask]
                )
        history['per_class_acc'].append(per_class)

        scheduler.step(current_acc)

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:>3}/{EPOCHS}] | Loss: {avg_loss:.4f} | Acc: {current_acc*100:.1f}% | LR: {lr_now:.6f}")
            for g in ['swipe_left', 'swipe_right']:
                if g in per_class:
                    print(f"  {g}: {per_class[g]*100:.1f}%")

        if current_acc > best_acc:
            best_acc      = current_acc
            patience_left = PATIENCE
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_NAME))
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\nEarly stopping на епосі {epoch+1} (не покращилось {PATIENCE} епох)")
                break

    print(f"\nНавчання завершено! Найкраща точність: {best_acc*100:.2f}%")

    # 5. Фінальна per-class точність
    print("\n--- Фінальна точність по класах ---")
    final_per_class = history['per_class_acc'][-1]
    for gesture in GESTURES:
        acc = final_per_class.get(gesture, 0)
        bar = "█" * int(acc * 20)
        status = "✓" if acc >= 0.85 else "⚠"
        print(f"  {status} {gesture:<15} {acc*100:>5.1f}%  {bar}")

    # 6. Графіки
    _plot_history(history)
    _plot_confusion_matrix(all_labels, all_preds)

    print(f"\nГрафіки збережені в: {DIAGRAM_DIR}/")


# ---------------------------------------------------------------------------
# Візуалізація
# ---------------------------------------------------------------------------

def _plot_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Loss', color='#e74c3c')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['test_acc'], label='Overall Acc', color='#2ecc71', linewidth=2)

    # Окремо свайпи
    for gesture, color in [('swipe_left', '#3498db'), ('swipe_right', '#9b59b6')]:
        vals = [ep.get(gesture, 0) for ep in history['per_class_acc']]
        axes[1].plot(vals, label=gesture, color=color, linestyle='--', alpha=0.8)

    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'learning_process.png'), dpi=120)
    plt.close()


def _plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)

    # Нормалізована матриця (відсотки) — краще бачити плутанину
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=GESTURES, yticklabels=GESTURES,
                cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_norm, annot=True, fmt='.0%',
                xticklabels=GESTURES, yticklabels=GESTURES,
                cmap='Blues', vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title('Confusion Matrix (normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'confusion_matrix.png'), dpi=120)
    plt.close()
    print("Confusion matrix збережена.")


if __name__ == "__main__":
    train_model()