"""
Phase 0 diagnostic step 4: the single most diagnostic check in the
protocol. Train the current architecture on a tiny, well-separated
subset for many epochs at a higher learning rate.

  - Overfits cleanly -> the bug is optimization-scale (LR/epochs on
    the full dataset), not expressivity.
  - Cannot overfit even 40 points, despite gradients/optimizer already
    confirmed healthy (steps 1-2) -> a real expressivity/observable
    bottleneck -> proceed to the observable ablation (step 5).

See notebook/04D_phase0_diagnostics.ipynb.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def select_tiny_subset(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically select n_per_class samples of each binary
    class (0 and 1), shuffled together."""

    rng = np.random.default_rng(seed)
    y = np.asarray(y).ravel()

    indices = []

    for cls in (0, 1):
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=n_per_class, replace=False)
        indices.extend(chosen.tolist())

    indices = np.array(indices)
    rng.shuffle(indices)

    return X[indices], y[indices]


def run_overfit_test(
    model: nn.Module,
    X_small: np.ndarray,
    y_small: np.ndarray,
    epochs: int = 200,
    lr: float = 0.05,
) -> List[dict]:
    """
    Full-batch gradient descent on a tiny subset.

    Returns
    -------
    list[dict]
        One entry per epoch: epoch, loss, accuracy.
    """

    x_tensor = torch.tensor(X_small, dtype=torch.float32)
    y_tensor = torch.tensor(y_small, dtype=torch.float32).reshape(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            accuracy = (predictions == y_tensor).float().mean().item()

        history.append({
            "epoch": epoch,
            "loss": loss.item(),
            "accuracy": accuracy,
        })

    return history
