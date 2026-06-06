"""Inference utilities for trained LSTM models."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from ola_lstm.config import Config
from ola_lstm.data import DAY_NAMES, MONTH_NAMES, build_synthetic_sequence
from ola_lstm.metrics import demand_level


@dataclass
class PredictionResult:
    ride_count: int
    demand_level: str
    hour: int
    day_of_week: int
    month: int
    day_name: str
    month_name: str
    next_6_hours: list[tuple[int, int]]


class RideDemandPredictor:
    """Load a trained model and scaler for ride demand inference."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.model = load_model(self.config.model_path)
        with open(self.config.scaler_path, "rb") as f:
            self.scaler: MinMaxScaler = pickle.load(f)

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path | str) -> RideDemandPredictor:
        config = Config(output_dir=Path(artifacts_dir))
        return cls(config)

    def _predict_sequence(self, sequence: np.ndarray) -> int:
        scaled = self.scaler.transform(sequence.reshape(-1, 1)).flatten()
        x_input = scaled.reshape(1, self.config.seq_len, 1)
        pred_scaled = self.model.predict(x_input, verbose=0)[0][0]
        return int(self.scaler.inverse_transform([[pred_scaled]])[0][0])

    def predict(
        self,
        hour: int,
        day_of_week: int,
        month: int,
        seed: int | None = None,
    ) -> PredictionResult:
        """Predict ride demand for a given hour/day/month context."""
        seq = build_synthetic_sequence(
            hour, day_of_week, month, self.config.seq_len, seed=seed
        )
        count = self._predict_sequence(seq)

        next_6: list[tuple[int, int]] = []
        for i in range(6):
            nh = (hour + i + 1) % 24
            seq_h = build_synthetic_sequence(
                nh, day_of_week, month, self.config.seq_len, seed=(seed or 0) + i + 1
            )
            next_6.append((nh, self._predict_sequence(seq_h)))

        return PredictionResult(
            ride_count=count,
            demand_level=demand_level(count),
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            day_name=DAY_NAMES[day_of_week],
            month_name=MONTH_NAMES[month - 1],
            next_6_hours=next_6,
        )
