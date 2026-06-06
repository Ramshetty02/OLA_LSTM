"""Tests for evaluation metrics."""

import numpy as np

from ola_lstm.metrics import compute_metrics, demand_level


def test_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    metrics = compute_metrics(y, y)
    assert metrics.mse == 0.0
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.mape == 0.0


def test_demand_level_tiers():
    assert demand_level(10) == "Low"
    assert demand_level(25) == "Moderate"
    assert demand_level(50) == "High (Surge likely)"
