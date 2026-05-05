import torch
import torch.nn as nn


class GestureLSTM(nn.Module):
    """
    Двошаровий LSTM класифікатор жестів.

    Вхід:  (batch, seq_length, input_size)
    Вихід: (batch, num_classes) — логіти (без softmax)

    input_size = 252 при USE_DELTA=True (координати + дельта руху)
               = 126 при USE_DELTA=False (тільки координати)
    """

    def __init__(self, input_size: int | None = None,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int | None = None,
                 dropout: float = 0.3):
        super().__init__()

        # Завантажуємо з config якщо не передано явно
        if input_size is None:
            from src.config import INPUT_SIZE
            input_size = INPUT_SIZE

        if num_classes is None:
            from src.config import GESTURES
            num_classes = len(GESTURES)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (batch, seq, hidden)
        out = self.dropout(out[:, -1, :])  # останній крок + dropout
        return self.fc(out)             # (batch, num_classes)