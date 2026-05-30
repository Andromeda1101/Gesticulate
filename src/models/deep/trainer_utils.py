"""Shared PyTorch training utilities for deep baselines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


def train_torch_classifier(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 5,
    device: str | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Train *model* with early stopping on validation accuracy."""
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = -1.0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    fit_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += len(yb)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += len(yb)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    fit_seconds = time.perf_counter() - fit_start
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "fit_seconds": fit_seconds,
        "best_val_accuracy": best_val_acc,
        "epochs_run": len(history),
        "history": history,
        "device": dev,
    }


def predict_torch_classifier(
    model: Any,
    loader: DataLoader,
    *,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_indices, y_proba) for a trained torch model."""
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(dev)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_probs.append(probs)
    return np.array(all_preds, dtype=np.int64), np.vstack(all_probs)
