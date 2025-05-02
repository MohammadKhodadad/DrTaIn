#!/usr/bin/env python
# run_experiment.py
"""
End‑to‑end pipeline:
    data → model → training → evaluation → visualisation.
"""

from pathlib import Path
import argparse
import numpy as np
import torch

from utils.chemberta_exp.chemberta_model import ChemBERTaRegressor
from utils.chemberta_exp.affinity_data   import get_dataloaders
from utils.chemberta_exp.trainer         import train_affinity
from utils.chemberta_exp.visualization   import (
    parity_plot,
    distribution_plot,
    residual_hist,
    residuals_plot,
    history_curves,
)

# ────────────────────────────────────────────────────────────────────────────
def evaluate_preds(model, loader, scaler, device):
    """Return inverse‑scaled (y_true, y_pred) for a loader."""
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask = ids.to(device), mask.to(device)
            pred = model(ids, mask).cpu().numpy()
            y_pred_all.append(pred)
            y_true_all.append(y.numpy())

    y_pred_scaled = np.concatenate(y_pred_all).reshape(-1, 1)
    y_true_scaled = np.concatenate(y_true_all).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_true_scaled).flatten()
    return y_true, y_pred


# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with SMILES + target col")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # 1) data ────────────────────────────────────────────────────────
    print("Loading Data ...")
    train_loader, test_loader, scaler = get_dataloaders(
        args.csv, batch_size=args.batch_size
    )

    # carve 10 % of train → validation
    val_frac = 0.1
    full_train_ds = train_loader.dataset
    val_size = int(len(full_train_ds) * val_frac)
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    # 2) model ───────────────────────────────────────────────────────
    print("Loading Model ...")
    model = ChemBERTaRegressor()

    # 3) training / validation / test ───────────────────────────────
    print("Training ...")
    history, best_state, test_metrics = train_affinity(
        model,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        epochs=args.epochs,
        lr=args.lr,
        save_path=out_dir / "best_model.pt",
    )

    # 4) evaluation for visualisations ───────────────────────────────
    print("Evaluation and Visualization ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(best_state)
    model.to(device)

    y_true, y_pred = evaluate_preds(model, test_loader, scaler, device)

    # 5) plots ───────────────────────────────────────────────────────
    parity_plot(
        y_true,
        y_pred,
        title="Test Parity",
        save_path=out_dir / "parity_test.png",
        show=False,
    )
    distribution_plot(
        y_true,
        y_pred,
        title="Distribution – Test",
        save_path=out_dir / "distribution_test.png",
        show=False,
    )
    residual_hist(
        y_true,
        y_pred,
        title="Residuals Histogram – Test",
        save_path=out_dir / "residual_hist_test.png",
        show=False,
    )
    residuals_plot(
        y_true,
        y_pred,
        title="Residuals vs Predicted – Test",
        save_path=out_dir / "residuals_vs_pred_test.png",
        show=False,
    )
    history_curves(
        history,
        metric="loss",
        title="Training vs Validation Loss",
        save_path=out_dir / "loss_curves.png",
        show=False,
    )

    # 6) console summary ────────────────────────────────────────────
    print(
        f"\nFinished. Test R² = {test_metrics['r2']:.4f}, "
        f"loss = {test_metrics['loss']:.4f}. "
        f"Plots + weights saved in: {out_dir.resolve()}"
    )


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
