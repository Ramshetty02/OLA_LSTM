"""Visualization helpers for EDA and model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"primary": "#185FA5", "accent": "#D85A30", "success": "#1D9E75"}


def plot_eda(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Four-panel exploratory analysis of ride demand patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Ola Bike Ride Demand — Exploratory Analysis", fontsize=14, fontweight="bold")

    week_data = df["ride_count"][:168]
    axes[0, 0].plot(week_data.values, color=COLORS["primary"], linewidth=1.2)
    axes[0, 0].set_title("One Week of Ride Demand (Hourly)")
    axes[0, 0].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Ride Count")
    axes[0, 0].set_xticks(range(0, 168, 24))
    axes[0, 0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0, 0].grid(True, alpha=0.3)

    hourly_avg = df.groupby("hour")["ride_count"].mean()
    axes[0, 1].bar(hourly_avg.index, hourly_avg.values, color=COLORS["primary"], alpha=0.8)
    axes[0, 1].set_title("Average Ride Demand by Hour of Day")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Avg Ride Count")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    daily_avg = df.groupby("day_of_week")["ride_count"].mean()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1, 0].bar(days, daily_avg.values, color=COLORS["accent"], alpha=0.8)
    axes[1, 0].set_title("Average Ride Demand by Day of Week")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Avg Ride Count")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    monthly_avg = df.groupby("month")["ride_count"].mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[1, 1].bar(months, monthly_avg.values, color=COLORS["success"], alpha=0.8)
    axes[1, 1].set_title("Average Ride Demand by Month (Seasonal)")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Avg Ride Count")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_loss(history, output_path: Path | None = None) -> None:
    """Plot train vs validation MSE over epochs."""
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss", color=COLORS["primary"], linewidth=1.5)
    plt.plot(
        history.history["val_loss"],
        label="Val Loss",
        color=COLORS["accent"],
        linewidth=1.5,
        linestyle="--",
    )
    plt.title("Training & Validation Loss (MSE) over Epochs", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_n: int = 168,
    output_path: Path | None = None,
) -> None:
    """Compare actual vs predicted ride demand on the test set."""
    plt.figure(figsize=(14, 5))
    plt.plot(y_true[:plot_n], label="Actual", color=COLORS["primary"], linewidth=1.5)
    plt.plot(
        y_pred[:plot_n],
        label="LSTM Predicted",
        color=COLORS["accent"],
        linewidth=1.5,
        linestyle="--",
    )
    plt.title("Actual vs LSTM Predicted Ride Demand (Test Set — 7 Days)", fontweight="bold")
    plt.xlabel("Hour")
    plt.ylabel("Ride Count")
    plt.xticks(range(0, plot_n, 24), [f"Day {i + 1}" for i in range(7)])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
