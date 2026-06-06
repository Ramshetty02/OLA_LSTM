"""Tests for LSTM model construction."""

from ola_lstm.config import Config
from ola_lstm.model import build_lstm_model


def test_build_lstm_model_output_shape():
    config = Config(seq_len=24)
    model = build_lstm_model(config)
    assert model.input_shape == (None, 24, 1)
    assert model.output_shape == (None, 1)
