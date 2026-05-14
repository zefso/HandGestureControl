import os
import json
import torch

# ---------------------------------------------------------------------------
# Корінь проекту — не залежить від робочого каталогу при запуску
# ---------------------------------------------------------------------------
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC        = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SRC, 'gestures.json')


def _load() -> dict:
    try:
        with open(_CONFIG_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[config] gestures.json не знайдено або пошкоджено: {e}")
        return {}


_cfg       = _load()
_model_cfg = _cfg.get("model", {})
_mouse_cfg = _cfg.get("mouse", {})
_vol_cfg   = _cfg.get("volume", {})

# ---------------------------------------------------------------------------
# Жести і профілі
# ---------------------------------------------------------------------------
GESTURES        = _cfg.get("gestures", ['static', 'swipe_right', 'swipe_left', 'ok', 'stop', 'browser', 'fist_left'])
GESTURE_ACTIONS = _cfg.get("profiles", {}).get(_cfg.get("active_profile", "default"), {}).get("actions", {})

# ---------------------------------------------------------------------------
# Розмір вхідних даних
#   USE_DELTA=True  → 252 ознаки (координати + дельта між кадрами) — РЕКОМЕНДОВАНО
#   USE_DELTA=False → 126 ознак (лише координати, зворотна сумісність)
#
# ⚠️  При зміні USE_DELTA ОБОВ'ЯЗКОВО:
#     1. python src/smart_collector.py --clean  (очистити старий датасет)
#     2. Зібрати новий датасет
#     3. python src/train.py
# ---------------------------------------------------------------------------
USE_DELTA  = True
INPUT_SIZE = 252 if USE_DELTA else 126

# ---------------------------------------------------------------------------
# Параметри моделі / інференсу
# ---------------------------------------------------------------------------
SEQ_LENGTH      = _model_cfg.get("seq_length",      30)
THRESHOLD       = _model_cfg.get("threshold",        0.92)
SWITCH_FRAMES   = _model_cfg.get("switch_frames",    7)
COOLDOWN_FRAMES       = _model_cfg.get("cooldown_frames",       40)
SWIPE_COOLDOWN_FRAMES = _model_cfg.get("swipe_cooldown_frames", 12)  # Короткий cooldown для swipe_left / swipe_right

# ---------------------------------------------------------------------------
# Параметри миші
# ---------------------------------------------------------------------------
MOUSE_SMOOTHING = _mouse_cfg.get("smoothing",    0.3)
MOUSE_ZONE_X    = _mouse_cfg.get("click_zone_x", [0.25, 0.75])
MOUSE_ZONE_Y    = _mouse_cfg.get("click_zone_y", [0.25, 0.65])

# ---------------------------------------------------------------------------
# Параметри гучності
# ---------------------------------------------------------------------------
VOLUME_SMOOTHING    = _vol_cfg.get("smoothing",     0.15)
VOLUME_DISTANCE_MIN = _vol_cfg.get("distance_min",  0.02)
VOLUME_DISTANCE_MAX = _vol_cfg.get("distance_max",  0.20)

# ---------------------------------------------------------------------------
# Шляхи (абсолютні — не залежать від CWD)
# ---------------------------------------------------------------------------
DATA_PATH   = os.path.join(_ROOT, 'data')
MODEL_DIR   = os.path.join(_ROOT, 'models')
DIAGRAM_DIR = os.path.join(_ROOT, 'diagram')
MODEL_NAME  = 'gesture_lstm_best.pth'
MODEL_PATH  = os.path.join(MODEL_DIR, MODEL_NAME)

# ---------------------------------------------------------------------------
# Залізо
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Гіперпараметри навчання
# ---------------------------------------------------------------------------
BATCH_SIZE    = 32
LEARNING_RATE = 0.0005
EPOCHS        = 200
PATIENCE      = 20   # Early stopping: зупинитись після N епох без покращення
