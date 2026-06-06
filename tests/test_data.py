"""Tests for synthetic data generation."""

import pandas as pd

from ola_lstm.data import (
    SEASON_MULTIPLIERS,
    base_demand_for_hour,
    build_synthetic_sequence,
    generate_ola_dataset,
)


def test_generate_ola_dataset_shape_and_columns():
    df = generate_ola_dataset(start="2024-01-01", periods=100, seed=0)
    assert len(df) == 100
    assert isinstance(df.index, pd.DatetimeIndex)
    for col in ("ride_count", "hour", "day_of_week", "month", "is_weekend"):
        assert col in df.columns


def test_ride_counts_are_non_negative():
    df = generate_ola_dataset(periods=500, seed=1)
    assert (df["ride_count"] >= 0).all()


def test_base_demand_weekday_vs_weekend():
    weekday_morning = base_demand_for_hour(8, is_weekend=False)
    weekend_morning = base_demand_for_hour(8, is_weekend=True)
    assert weekday_morning > weekend_morning


def test_synthetic_sequence_length():
    seq = build_synthetic_sequence(hour=12, day_of_week=2, month=6, seq_len=24, seed=42)
    assert seq.shape == (24,)
    assert (seq >= 0).all()


def test_season_multipliers_cover_all_months():
    assert set(SEASON_MULTIPLIERS.keys()) == set(range(1, 13))
