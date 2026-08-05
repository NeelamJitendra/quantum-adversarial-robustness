"""
Train one VQC for a given seed, using a validated ExperimentConfig
(see src/config.py and configs/base_experiment.json -- defaults are
the Phase 0-accepted single_z/ansatz_reps=4/lr=0.05 configuration,
see notebook/04D_phase0_diagnostics.ipynb).

Trains once, noiselessly (per the thesis architecture: noise and
adversarial attacks are applied at inference time on this checkpoint,
not by retraining under each condition -- see the approved
architecture plan). Saves a best-val-loss checkpoint, a metadata JSON,
and a full per-epoch history JSON (written incrementally so progress
is never lost if a run is interrupted).

CLI usage:
    python -m src.experiments.train_seed --seed 42
    python -m src.experiments.train_seed --seed 42 --epochs 10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import ExperimentConfig
from src.evaluation.metrics import compute_metrics
from src.models.hybrid_classifier import HybridClassifier
from src.models.quantum_model import create_model, get_output_dim
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "binary"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
HISTORY_DIR = PROJECT_ROOT / "results" / "training_history"


def train_vqc_seed(seed: int, config: ExperimentConfig) -> dict:
    """
    Train one VQC for `seed` using `config.circuit`/`config.train`.

    Returns
    -------
    dict
        Summary row (seed, best_val_loss, test metrics, checkpoint path)
        -- also what run_all_seeds.py aggregates into a results table.
    """

    set_seed(seed)

    X_train_full = np.load(DATA_DIR / "X_train.npy")
    y_train_full = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=seed,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    quantum_model = create_model(
        num_qubits=config.circuit.num_qubits,
        feature_reps=config.circuit.feature_reps,
        ansatz_reps=config.circuit.ansatz_reps,
        observable_mode=config.circuit.observable_mode,
        seed=seed,
    )
    output_dim = get_output_dim(
        config.circuit.num_qubits, config.circuit.observable_mode
    )
    model = HybridClassifier(quantum_model, quantum_output_dim=output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    criterion = nn.BCEWithLogitsLoss()

    x_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    x_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    n = len(x_train_t)
    batch = config.train.batch_size

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"vqc_seed{seed}.pt"
    history_path = HISTORY_DIR / f"vqc_seed{seed}_history.json"

    history = []
    best_val_loss = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(config.train.epochs):
        perm = torch.randperm(n)
        model.train()
        total_loss = 0.0

        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb, yb = x_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_out = model(x_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            val_preds = (torch.sigmoid(val_out) >= 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_s": time.time() - t0,
        })

        history_path.write_text(
            json.dumps({"seed": seed, "history": history, "done": False}, indent=2)
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_out = model(x_test_t)
        test_preds = (torch.sigmoid(test_out) >= 0.5).float()

    metrics = compute_metrics(
        y_test_t.numpy().ravel(), test_preds.numpy().ravel()
    )
    metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    metadata = {
        "seed": seed,
        "circuit": config.circuit.__dict__,
        "train": config.train.__dict__,
        "best_val_loss": best_val_loss,
        "test_metrics": metrics,
    }
    (MODELS_DIR / f"vqc_seed{seed}_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    history_path.write_text(
        json.dumps(
            {"seed": seed, "history": history, "test_metrics": metrics, "done": True},
            indent=2,
        )
    )

    return {
        "seed": seed,
        "best_val_loss": best_val_loss,
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "base_experiment.json"),
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override config.train.epochs without editing the config file.",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    config.seed = args.seed
    if args.epochs is not None:
        config.train.epochs = args.epochs

    summary = train_vqc_seed(args.seed, config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
