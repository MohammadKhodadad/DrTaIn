# visualization.py
"""
Utility plots for affinity‑regression experiments.

Examples
--------
from visualization import (
    parity_plot,
    distribution_plot,
    residuals_plot,
    residual_hist,
    history_curves,
)

# After training …
parity_plot(y_true, y_pred, title="Test set")
distribution_plot(y_true, y_pred)
residuals_plot(y_true, y_pred)
history_curves(history, metric="loss")      # history = list of dicts from trainer
"""
from typing import List, Sequence, Optional, Mapping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ────────────────────────────────────────────────────────────────────────────
def _maybe_save(fig: mpl.figure.Figure, save_path: Optional[str], show: bool):
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
def parity_plot(
    actual: Sequence[float],
    pred: Sequence[float],
    *,
    title: str = "Parity Plot",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Scatter plot with y=x reference line and R² in title."""
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        actual, pred,
        c=np.abs(actual - pred),
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
    )
    ax.plot([actual.min(), actual.max()],
            [actual.min(), actual.max()],
            ls="--", c="red", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title} (R² = {r2:.4f})")
    fig.colorbar(scatter, label="Absolute error")
    ax.grid(True)

    _maybe_save(fig, save_path, show)


# ────────────────────────────────────────────────────────────────────────────
def distribution_plot(
    actual: Sequence[float],
    pred: Sequence[float],
    *,
    bins: int | Sequence[int] = 30,
    title: str = "Actual vs Predicted Distributions",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Overlayed histograms of actual vs predicted values."""
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    bins = np.histogram_bin_edges(np.concatenate([actual, pred]), bins=bins)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(actual, bins=bins, alpha=0.5, label="Actual")
    ax.hist(pred,   bins=bins, alpha=0.5, label="Predicted")
    ax.set_xlabel("Affinity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    _maybe_save(fig, save_path, show)


# ────────────────────────────────────────────────────────────────────────────
def residual_hist(
    actual: Sequence[float],
    pred: Sequence[float],
    *,
    bins: int | Sequence[int] = 30,
    title: str = "Residual Histogram",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Histogram of residuals (actual − predicted)."""
    res = np.asarray(actual) - np.asarray(pred)
    bins = np.histogram_bin_edges(res, bins=bins)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(res, bins=bins, alpha=0.7)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True)

    _maybe_save(fig, save_path, show)


def residuals_plot(
    actual: Sequence[float],
    pred: Sequence[float],
    *,
    title: str = "Residuals vs Predicted",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Scatter of residuals against predictions."""
    pred = np.asarray(pred)
    res = np.asarray(actual) - pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred, res, alpha=0.7)
    ax.axhline(0, ls="--", c="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    ax.grid(True)

    _maybe_save(fig, save_path, show)


# ────────────────────────────────────────────────────────────────────────────
def history_curves(
    history: List[Mapping[str, float]],
    *,
    metric: str = "loss",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot training/validation curves across epochs.

    Parameters
    ----------
    history : list of dicts (output from trainer.train_affinity)
              Each dict should at least have keys 'epoch', 'train_{metric}',
              'val_{metric}'  (e.g. 'train_loss', 'val_loss').
    metric  : 'loss' or 'r2' (or any suffix present in history keys)
    """
    epochs = [h["epoch"] for h in history]
    train_vals = [h[f"train_{metric}"] for h in history]
    val_vals = [h[f"val_{metric}"]   for h in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_vals, label=f"Train {metric}")
    ax.plot(epochs, val_vals,   label=f"Val {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Training history – {metric.upper()}")
    ax.legend()
    ax.grid(True)

    _maybe_save(fig, save_path, show)
