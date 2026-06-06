"""Evaluation metrics for ride demand forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastMetrics:
    mse: float
    rmse: float
    mae: float
    mape: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"MSE={self.mse:.4f}  RMSE={self.rmse:.4f}  "
            f"MAE={self.mae:.4f}  MAPE={self.mape:.2f}%"
        )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ForecastMetrics:
    """Compute regression metrics on inverse-transformed predictions."""
    mse = float(mean_squared_error(y_true, y_pred))
    return ForecastMetrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
    )


def demand_level(count: float) -> str:
    """Map predicted ride count to a human-readable demand tier."""
    if count < 15:
        return "Low"
    if count < 40:
        return "Moderate"
    return "High (Surge likely)"
