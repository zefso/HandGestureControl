import os
import torch
# 1. СПИСОК ЖЕСТІВ (Додавай нові назви сюди)
GESTURES = ['static', 'swipe_right', 'swipe_left', 'ok', 'stop', 'browser', 'fist_left']

# 2. ПАРАМЕТРИ МОДЕЛІ
SEQ_LENGTH = 30
INPUT_SIZE = 126  # 21 точка * 3 коорд * 2 руки

# 3. ШЛЯХИ
DATA_PATH = os.path.join('data')
MODEL_PATH = os.path.join('models', 'gesture_lstm_best.pth')

# 4. НАЛАШТУВАННЯ ІНТЕРФЕЙСУ
THRESHOLD = 0.92


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- КОНФІГУРАЦІЯ ---
MODEL_DIR = 'models'
DIAGRAM_DIR = 'diagram'
MODEL_NAME = 'gesture_lstm_best.pth'
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 150
SWITCH_FRAMES = 7

