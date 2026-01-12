import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import GestureLSTM

DATA_PATH = 'data'
gestures = ['static', 'swipe_right']
sequence_length = 30

def train():
    X, y = [], []
    for idx, gesture in enumerate(gestures):
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_path):
            continue
            
        for seq in os.listdir(gesture_path):
            window = []
            for frame_num in range(sequence_length):
                file_path = os.path.join(gesture_path, seq, f"{frame_num}.npy")
                if os.path.exists(file_path):
                    res = np.load(file_path)
                    window.append(res)
            
            if len(window) == sequence_length:
                X.append(window)
                y.append(idx)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)

    epochs = 100 
    model = GestureLSTM(num_classes=len(gestures))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 

    print(f"Дані завантажено: {len(X)} зразків.")
    print("Починаємо навчання...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            _, predicted = torch.max(outputs.data, 1)
            total = y.size(0)
            correct = (predicted == y).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/test_model.pth')
    print("\nМодель збережена успішно!")

if __name__ == "__main__":
    train()