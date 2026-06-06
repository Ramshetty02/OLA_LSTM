"""Tests for feature engineering."""

import numpy as np

from ola_lstm.config import Config
from ola_lstm.features import create_sequences, prepare_dataset


def test_create_sequences_shape():
    data = np.arange(30, dtype=float)
    X, y = create_sequences(data, seq_len=5)
    assert X.shape == (25, 5)
    assert y.shape == (25,)
    assert X[0].tolist() == [0, 1, 2, 3, 4]
    assert y[0] == 5


def test_prepare_dataset_split_sizes():
    data = np.random.default_rng(0).integers(0, 100, size=1000).astype(float)
    config = Config(seq_len=24, train_ratio=0.7, val_ratio=0.15)
    split = prepare_dataset(data, config)

    total = len(split.X_train) + len(split.X_val) + len(split.X_test)
    assert total == 1000 - config.seq_len
    assert split.X_train.shape[1:] == (config.seq_len, 1)
    assert split.scaler is not None
