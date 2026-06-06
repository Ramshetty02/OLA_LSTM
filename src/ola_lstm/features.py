"""Feature engineering: scaling, sequences, and train/val/test splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ola_lstm.config import Config


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler


def create_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of length seq_len to produce (X, y) pairs."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def prepare_dataset(ride_counts: np.ndarray, config: Config) -> DatasetSplit:
    """Scale ride counts, build sequences, and split into train/val/test."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(ride_counts.reshape(-1, 1)).flatten()

    X, y = create_sequences(scaled, config.seq_len)
    n = len(X)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    def reshape(split_x: np.ndarray) -> np.ndarray:
        return split_x.reshape(-1, config.seq_len, 1)

    return DatasetSplit(
        X_train=reshape(X[:train_end]),
        y_train=y[:train_end],
        X_val=reshape(X[train_end:val_end]),
        y_val=y[train_end:val_end],
        X_test=reshape(X[val_end:]),
        y_test=y[val_end:],
        scaler=scaler,
    )
