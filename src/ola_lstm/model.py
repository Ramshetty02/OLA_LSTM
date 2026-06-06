"""LSTM model definition."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from ola_lstm.config import Config


def build_lstm_model(config: Config) -> Sequential:
    """Build a stacked LSTM regressor for hourly ride demand."""
    model = Sequential(
        [
            LSTM(
                config.lstm_units_1,
                input_shape=(config.seq_len, 1),
                return_sequences=True,
            ),
            Dropout(config.dropout),
            LSTM(config.lstm_units_2, return_sequences=False),
            Dropout(config.dropout),
            Dense(config.dense_units, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name="ola_lstm_forecaster",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
