"""Command-line interface for training and inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ola_lstm.config import Config
from ola_lstm.predict import RideDemandPredictor
from ola_lstm.train import train_model


def _train_cmd(args: argparse.Namespace) -> int:
    config = Config(
        output_dir=Path(args.output),
        epochs=args.epochs,
        random_seed=args.seed,
    )
    result = train_model(config)
    metrics = result["metrics"]
    print("\nTraining complete.")
    print(f"Artifacts saved to: {config.output_dir.resolve()}")
    print(f"Test metrics: {metrics}")
    return 0


def _predict_cmd(args: argparse.Namespace) -> int:
    predictor = RideDemandPredictor.from_artifacts(args.artifacts)
    result = predictor.predict(args.hour, args.day, args.month, seed=args.seed)
    print("=" * 45)
    print("       LSTM RIDE DEMAND PREDICTION")
    print("=" * 45)
    print(f"  Time   : {result.hour:02d}:00  |  {result.day_name}")
    print(f"  Month  : {result.month_name}")
    print(f"  Rides  : {result.ride_count} requests/hour")
    print(f"  Level  : {result.demand_level}")
    print("=" * 45)
    print("\n6-hour ahead forecast:")
    for h, count in result.next_6_hours:
        print(f"  {h:02d}:00 → {count} rides")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ola-lstm",
        description="Ola bike ride demand forecasting with stacked LSTM",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the LSTM model and save artifacts")
    train.add_argument("--output", default="artifacts", help="Output directory")
    train.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    train.add_argument("--seed", type=int, default=42, help="Random seed")
    train.set_defaults(func=_train_cmd)

    predict = sub.add_parser("predict", help="Run inference with a trained model")
    predict.add_argument("--artifacts", default="artifacts", help="Artifacts directory")
    predict.add_argument("--hour", type=int, default=8, choices=range(24), help="Hour (0-23)")
    predict.add_argument("--day", type=int, default=0, choices=range(7), help="Day of week (0=Mon)")
    predict.add_argument("--month", type=int, default=6, choices=range(1, 13), help="Month (1-12)")
    predict.add_argument("--seed", type=int, default=42, help="Random seed for demo sequence")
    predict.set_defaults(func=_predict_cmd)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
