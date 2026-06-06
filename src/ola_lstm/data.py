"""Synthetic Ola bike ride demand data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON_MULTIPLIERS: dict[int, float] = {
    12: 0.85,
    1: 0.85,
    2: 0.90,
    3: 1.00,
    4: 1.05,
    5: 1.10,
    6: 1.15,
    7: 1.10,
    8: 1.05,
    9: 0.80,
    10: 0.75,
    11: 0.80,
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def base_demand_for_hour(hour: int, is_weekend: bool) -> int:
    """Return expected base ride count for a given hour."""
    if 0 <= hour <= 5:
        return 6
    if 6 <= hour <= 9:
        return 25 if is_weekend else 55
    if 10 <= hour <= 16:
        return 35 if is_weekend else 30
    if 17 <= hour <= 20:
        return 42 if is_weekend else 62
    return 18


def generate_ola_dataset(
    start: str = "2023-01-01",
    periods: int = 8760,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly Ola bike ride demand data.

    Returns a DataFrame indexed by timestamp with ride_count and calendar features.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    ride_counts: list[int] = []

    for ts in timestamps:
        hour = ts.hour
        day = ts.dayofweek
        month = ts.month
        is_weekend = day >= 5
        season_mult = SEASON_MULTIPLIERS[month]
        base = base_demand_for_hour(hour, is_weekend)
        noise = rng.normal(0, 4)
        ride_counts.append(max(0, int(base * season_mult + noise)))

    df = pd.DataFrame({"timestamp": timestamps, "ride_count": ride_counts})
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df.set_index("timestamp")


def build_synthetic_sequence(
    hour: int,
    day_of_week: int,
    month: int,
    seq_len: int = 24,
    seed: int | None = None,
) -> np.ndarray:
    """Build a 24-hour lookback sequence for interactive inference demos."""
    rng = np.random.default_rng(seed)
    is_weekend = day_of_week >= 5
    season_mult = SEASON_MULTIPLIERS[month]

    sequence = []
    for i in range(seq_len):
        h = (hour - seq_len + i) % 24
        base = base_demand_for_hour(h, is_weekend)
        noise = rng.normal(0, 2)
        sequence.append(max(0, base * season_mult + noise))

    return np.array(sequence, dtype=np.float32)
