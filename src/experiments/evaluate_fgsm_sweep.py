"""
FGSM epsilon-sweep pipeline: loads each trained VQC checkpoint and
attacks it via ART's FastGradientMethod, at inference time only (the
checkpoint is not retrained -- same "train once, evaluate under many
conditions" design as evaluate_noise_sweep.py).

Unlike the noise sweep (pure forward-pass shot sampling, ~6-9ms/sample,
cheap enough for the full 2956-sample test set), FGSM requires a
backward pass through the quantum layer for the input gradient
(~170ms/sample-epsilon, empirically calibrated -- see
notebook/07_fgsm_attack.ipynb), ~19x more expensive. Per the thesis
architecture's own compute-budget guidance, this runs on a stratified
SUBSAMPLE of the test set (500 of 2956, class-stratified,
seed-independent so every seed attacks the same set of points) rather
than the full test set.

CLI usage:
    python -m src.experiments.evaluate_fgsm_sweep --seeds 42,43,44,45,46,47,48,49
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.attacks.art_wrapper import build_art_classifier
from src.attacks.fgsm import run_fgsm_sweep
from src.config import CircuitConfig
from src.models.hybrid_classifier import HybridClassifier
from src.models.quantum_model import create_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "binary"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
SWEEP_DIR = PROJECT_ROOT / "results" / "fgsm_sweep"

EPSILONS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
SUBSAMPLE_SIZE = 500
SUBSAMPLE_SEED = 123  # fixed, independent of model seed -- every seed attacks the same points


def select_subsample(X_test: np.ndarray, y_test: np.ndarray, n: int = SUBSAMPLE_SIZE):
    """Class-stratified subsample of the test set, fixed across all
    model seeds so results are directly comparable seed-to-seed."""

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    y_test = y_test.ravel()

    indices = []
    for cls in np.unique(y_test):
        cls_indices = np.where(y_test == cls)[0]
        n_cls = round(n * len(cls_indices) / len(y_test))
        indices.extend(rng.choice(cls_indices, size=n_cls, replace=False).tolist())

    indices = np.array(indices)
    rng.shuffle(indices)

    return X_test[indices], y_test[indices]


def load_model(seed: int, circuit_config: CircuitConfig) -> HybridClassifier:
    quantum_model = create_model(
        observable_mode=circuit_config.observable_mode,
        ansatz_reps=circuit_config.ansatz_reps,
        feature_reps=circuit_config.feature_reps,
        num_qubits=circuit_config.num_qubits,
        seed=seed,
    )
    model = HybridClassifier(quantum_model, quantum_output_dim=1)
    model.load_state_dict(
        torch.load(MODELS_DIR / f"vqc_seed{seed}.pt", weights_only=True)
    )
    model.eval()
    return model


def run_fgsm_sweep_for_seed(seed: int, circuit_config: CircuitConfig) -> pd.DataFrame:
    X_train_full = np.load(DATA_DIR / "X_train.npy")
    y_train_full = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    X_train, _, _, _ = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=seed,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_sub, y_sub = select_subsample(X_test_s, y_test)
    X_sub = X_sub.astype(np.float32)
    y_sub = y_sub.astype(np.int64)

    model = load_model(seed, circuit_config)
    clip_values = (float(X_train_s.min()), float(X_train_s.max()))
    art_classifier = build_art_classifier(model, clip_values=clip_values, input_shape=(4,))

    df = run_fgsm_sweep(art_classifier, X_sub, y_sub, EPSILONS)
    df.insert(0, "seed", seed)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / f"fgsm_sweep_seed{seed}.csv", index=False)

    for _, row in df.iterrows():
        print(row.to_dict(), flush=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "base_experiment.json"),
    )
    args = parser.parse_args()

    from src.config import ExperimentConfig
    base_config = ExperimentConfig.from_json(args.config)

    seeds = [int(s) for s in args.seeds.split(",")]
    all_dfs = []

    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        all_dfs.append(run_fgsm_sweep_for_seed(seed, base_config.circuit))

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(SWEEP_DIR / "fgsm_sweep_all_seeds.csv", index=False)
    print(f"\nWritten to {SWEEP_DIR / 'fgsm_sweep_all_seeds.csv'}")


if __name__ == "__main__":
    main()
