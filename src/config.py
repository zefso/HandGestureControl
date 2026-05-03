import os
import json
import torch

# ---------------------------------------------------------------------------
# Завантаження з gestures.json
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'gestures.json')

def _load_json_config() -> dict:
    """Читає gestures.json. Якщо файл відсутній — повертає порожній dict."""
    try:
        with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[config] gestures.json не знайдено, використовуються дефолти.")
        return {}
    except json.JSONDecodeError as e:
        print(f"[config] Помилка парсингу gestures.json: {e}. Використовуються дефолти.")
        return {}

_cfg = _load_json_config()

# ---------------------------------------------------------------------------
# Жести — читаємо з JSON, fallback на захардкоджений список
# ---------------------------------------------------------------------------

GESTURES: list[str] = _cfg.get("gestures", [
    'static', 'swipe_right', 'swipe_left', 'ok', 'stop', 'browser', 'fist_left'
])

# Маппінг жест → дія (для hotkey / url / etc.)
GESTURE_ACTIONS: dict = _cfg.get("actions", {})

# ---------------------------------------------------------------------------
# Параметри моделі
# ---------------------------------------------------------------------------

_model_cfg = _cfg.get("model", {})

SEQ_LENGTH:     int   = _model_cfg.get("seq_length",     30)
THRESHOLD:      float = _model_cfg.get("threshold",      0.92)
SWITCH_FRAMES:  int   = _model_cfg.get("switch_frames",  7)
COOLDOWN_FRAMES: int  = _model_cfg.get("cooldown_frames", 40)

INPUT_SIZE: int = 126  # 21 точка * 3 координати * 2 руки — не змінюється

# ---------------------------------------------------------------------------
# Параметри миші
# ---------------------------------------------------------------------------

_mouse_cfg = _cfg.get("mouse", {})

MOUSE_SMOOTHING: float     = _mouse_cfg.get("smoothing", 0.3)
MOUSE_ZONE_X:    list[float] = _mouse_cfg.get("click_zone_x", [0.25, 0.75])
MOUSE_ZONE_Y:    list[float] = _mouse_cfg.get("click_zone_y", [0.25, 0.65])

# ---------------------------------------------------------------------------
# Параметри гучності
# ---------------------------------------------------------------------------

_vol_cfg = _cfg.get("volume", {})

VOLUME_SMOOTHING:     float = _vol_cfg.get("smoothing",     0.15)
VOLUME_DISTANCE_MIN:  float = _vol_cfg.get("distance_min",  0.02)
VOLUME_DISTANCE_MAX:  float = _vol_cfg.get("distance_max",  0.20)

# ---------------------------------------------------------------------------
# Шляхи
# ---------------------------------------------------------------------------

DATA_PATH:  str = os.path.join('data')
MODEL_DIR:  str = 'models'
DIAGRAM_DIR: str = 'diagram'
MODEL_NAME: str = 'gesture_lstm_best.pth'
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_NAME)

# ---------------------------------------------------------------------------
# Девайс і гіперпараметри тренування
# ---------------------------------------------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE:    int   = 32
LEARNING_RATE: float = 0.0005
EPOCHS:        int   = 150
