# trainer.py
"""
Generic training loop for ChemBERTa (or any PyTorch) regression model.

Example
-------
from chemberta_model import ChemBERTaRegressor
from affinity_data   import get_dataloaders
from trainer         import train_affinity

train_loader, test_loader, scaler = get_dataloaders("docked_leadlike.csv")
# split out 10 % of train for validation
val_size = int(0.1 * len(train_loader.dataset))
train_ds, val_ds = torch.utils.data.random_split(
    train_loader.dataset, [len(train_loader.dataset)-val_size, val_size]
)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=64, shuffle=False)

model = ChemBERTaRegressor()
history, best_state, test_metrics = train_affinity(
    model, train_loader, val_loader, test_loader, scaler,
    epochs=6, lr=1e-5, save_path="best_chemberta.pt"
)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score


# ─────────────────────────────────────────────────────────────────────────────
def _evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """Return (loss, R²) on a loader."""
    model.eval()
    total_loss, preds_all, y_all = 0.0, [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            pred = model(ids, mask)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * ids.size(0)
            preds_all.append(pred.cpu().numpy())
            y_all.append(y.cpu().numpy())

    preds = np.concatenate(preds_all).reshape(-1, 1)
    y_true = np.concatenate(y_all).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).flatten()
    y_inv = scaler.inverse_transform(y_true).flatten()

    return total_loss / len(loader.dataset), r2_score(y_inv, preds_inv)


# ─────────────────────────────────────────────────────────────────────────────
def train_affinity(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    test_loader:  torch.utils.data.DataLoader,
    scaler,
    *,
    device: Optional[torch.device] = None,
    epochs: int = 5,
    lr: float = 1e-5,
    loss_fn: nn.Module = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, float]], dict, Dict[str, float]]:
    """
    Train ‑> validate each epoch, keep best weights, evaluate on test.

    Returns
    -------
    history     : list of dicts with train/val loss + R² per epoch
    best_state  : state‑dict of the best validation model
    test_metrics: {"loss": float, "r2": float}
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = loss_fn or nn.SmoothL1Loss()
    optimiser = optim.AdamW(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []
    best_r2, best_state = -np.inf, None

    for epoch in range(1, epochs + 1):
        # ── training ───────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)

            optimiser.zero_grad()
            pred = model(ids, mask)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * ids.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ── validation ────────────────────────────────────────
        val_loss, val_r2 = _evaluate(model, val_loader, device, scaler, loss_fn)

        history.append(
            {"epoch": epoch, "train_loss": train_loss,
             "val_loss": val_loss, "val_r2": val_r2}
        )

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | val_R² {val_r2:.4f}"
            )

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── save best weights ─────────────────────────────────────
    if save_path and best_state is not None:
        torch.save(best_state, save_path)
        if verbose:
            print(f"Best model saved to {save_path} (val R² = {best_r2:.4f})")

    # ── test evaluation ───────────────────────────────────────
    model.load_state_dict(best_state)
    test_loss, test_r2 = _evaluate(model, test_loader, device, scaler, loss_fn)
    test_metrics = {"loss": test_loss, "r2": test_r2}
    if verbose:
        print(f"TEST | loss {test_loss:.4f} | R² {test_r2:.4f}")

    return history, best_state, test_metrics
