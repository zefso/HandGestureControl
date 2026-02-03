import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Додаємо шлях до моделі
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import GestureLSTM

# --- КОНФІГУРАЦІЯ ---
DATA_PATH = 'data'
MODEL_DIR = 'models'
DIAGRAM_DIR = 'diagram'
MODEL_NAME = 'gesture_lstm_best.pth'
SEQUENCE_LENGTH = 30
INPUT_SIZE = 126 # 21 точка * 3 координати * 2 руки
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 150

# Створення папок
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DIAGRAM_DIR, exist_ok=True)

class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_data(gestures):
    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(gestures)}
    
    for gesture in gestures:
        print(f"--> Завантаження жесту: {gesture}")
        gesture_path = os.path.join(DATA_PATH, gesture)
        sequences = os.listdir(gesture_path)
        
        for seq in sequences:
            seq_path = os.path.join(gesture_path, seq)
            # Сортування файлів за номером кадру
            frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.npy')], 
                          key=lambda x: int(x.split('.')[0]))
            
            window = []
            # Беремо перші 30 кадрів
            for frame in frames[:SEQUENCE_LENGTH]: 
                res = np.load(os.path.join(seq_path, frame))
                window.append(res)
            
            # Якщо кадрів менше 30 - заповнюємо нулями (Padding)
            while len(window) < SEQUENCE_LENGTH:
                window.append(np.zeros(INPUT_SIZE))
                
            X.append(window)
            y.append(label_map[gesture])
            
    return np.array(X), np.array(y)

def train_model():
    # 1. Підготовка даних
    gestures = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"Список жестів: {gestures}")
    
    X, y = load_data(gestures)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(GestureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(GestureDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Ініціалізація
    model = GestureLSTM(num_classes=len(gestures))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'test_acc': []}
    best_acc = 0.0
    
    # 3. Цикл навчання
    print(f"\nПочинаємо навчання на {EPOCHS} епох...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Валідація
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        current_acc = accuracy_score(all_labels, all_preds)
        avg_loss = running_loss / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(current_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Test Acc: {current_acc*100:.2f}%")
            
        # Збереження найкращої моделі
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_NAME))

    print(f"\nНавчання завершено! Найкраща точність: {best_acc*100:.2f}%")

    # 4. Візуалізація результатів
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Loss')
    plt.title('Історія помилки (Training Loss)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], label='Accuracy', color='orange')
    plt.title('Точність на тестах (Validation Accuracy)')
    plt.legend()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'learning_process.png'))

    # Матриця помилок
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=gestures, yticklabels=gestures, cmap='Blues')
    plt.title('Матриця помилок (Confusion Matrix)')
    plt.xlabel('Передбачено')
    plt.ylabel('Справжній жест')
    plt.savefig(os.path.join(DIAGRAM_DIR, 'final_confusion_matrix.png'))
    
    print(f"Усі графіки збережені в папці: {DIAGRAM_DIR}")

if __name__ == "__main__":
    train_model()