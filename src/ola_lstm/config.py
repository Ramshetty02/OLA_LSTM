"""Central configuration for training and inference."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Data
    start_date: str = "2023-01-01"
    periods: int = 8760  # 1 year of hourly data
    random_seed: int = 42

    # Features
    seq_len: int = 24
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Model
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dense_units: int = 32
    dropout: float = 0.2
    learning_rate: float = 0.001

    # Training
    epochs: int = 50
    batch_size: int = 32
    early_stop_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))
    model_filename: str = "ola_lstm_model.keras"
    scaler_filename: str = "scaler.pkl"
    metrics_filename: str = "metrics.json"

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.model_filename

    @property
    def scaler_path(self) -> Path:
        return self.output_dir / self.scaler_filename

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / self.metrics_filename

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio
