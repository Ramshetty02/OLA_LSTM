"""End-to-end training pipeline."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ola_lstm.config import Config
from ola_lstm.data import generate_ola_dataset
from ola_lstm.features import DatasetSplit, prepare_dataset
from ola_lstm.metrics import compute_metrics
from ola_lstm.model import build_lstm_model
from ola_lstm.viz import plot_eda, plot_predictions, plot_training_loss


def set_seeds(seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


def train_model(config: Config | None = None, save_artifacts: bool = True) -> dict:
    """
    Run the full training pipeline: data → features → train → evaluate → save.

    Returns a dict with metrics, history, model, and dataset split.
    """
    config = config or Config()
    set_seeds(config.random_seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_ola_dataset(
        start=config.start_date,
        periods=config.periods,
        seed=config.random_seed,
    )
    split = prepare_dataset(df["ride_count"].values, config)

    model = build_lstm_model(config)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
    ]

    history = model.fit(
        split.X_train,
        split.y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(split.X_val, split.y_val),
        callbacks=callbacks,
        verbose=1,
    )

    y_pred_scaled = model.predict(split.X_test, verbose=0).flatten()
    y_pred = split.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = split.scaler.inverse_transform(split.y_test.reshape(-1, 1)).flatten()
    metrics = compute_metrics(y_true, y_pred)

    if save_artifacts:
        _save_artifacts(config, model, split, metrics, history, df, y_true, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "split": split,
        "dataframe": df,
    }


def _save_artifacts(
    config: Config,
    model,
    split: DatasetSplit,
    metrics,
    history,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    model.save(config.model_path)
    with open(config.scaler_path, "wb") as f:
        pickle.dump(split.scaler, f)
    with open(config.metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    figures_dir = config.output_dir / "figures"
    plot_eda(df, figures_dir / "eda.png")
    plot_training_loss(history, figures_dir / "training_loss.png")
    plot_predictions(y_true, y_pred, output_path=figures_dir / "actual_vs_predicted.png")
