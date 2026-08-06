"""
PGD sweep for both the VQC and the classical baseline -- the
proposal's optional stretch goal ("PGD attacks may be included if time
permits").

Scope note: PGD costs ~180ms/sample-epsilon-iteration for the VQC
(empirically calibrated -- essentially the same per-step cost as
FGSM's single step, multiplied by max_iter iterations). Running at
FGSM's full scale (500 samples x 7 epsilons x 10 iterations) would
cost ~14 hours across 8 seeds. Since this is an optional extension
(unlike the FGSM sweep, which the proposal requires), this runs on a
smaller 50-sample subsample, 5 representative epsilons, and 7
iterations -- still enough to check whether the non-monotonic
attack-success pattern found with FGSM (07_fgsm_attack.ipynb) is an
artifact of FGSM's single linearized step, at a defensible compute
cost (~5 min/seed for the VQC; the classical baseline's PGD sweep is
far cheaper since it's a plain PyTorch backward pass, no quantum
circuit simulation).

CLI usage:
    python -m src.experiments.evaluate_pgd_sweep --seeds 42,43,44,45,46,47,48,49
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.attacks.art_wrapper import build_art_classifier
from src.attacks.pgd import run_pgd_sweep
from src.classical.baseline import ClassicalBaseline
from src.config import CircuitConfig
from src.experiments.evaluate_fgsm_sweep import load_model, select_subsample
from src.models.hybrid_classifier import HybridClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "binary"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
SWEEP_DIR = PROJECT_ROOT / "results" / "pgd_sweep"

PGD_SUBSAMPLE_SIZE = 50
PGD_EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.3]
PGD_MAX_ITER = 7


def _load_scaled_subsample(seed: int):
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

    X_sub, y_sub = select_subsample(X_test_s, y_test, n=PGD_SUBSAMPLE_SIZE)
    clip_values = (float(X_train_s.min()), float(X_train_s.max()))

    return X_sub.astype(np.float32), y_sub.astype(np.int64), clip_values


def run_pgd_sweep_for_seed_vqc(seed: int, circuit_config: CircuitConfig) -> pd.DataFrame:
    X_sub, y_sub, clip_values = _load_scaled_subsample(seed)

    model = load_model(seed, circuit_config)
    art_classifier = build_art_classifier(model, clip_values=clip_values, input_shape=(4,))

    df = run_pgd_sweep(art_classifier, X_sub, y_sub, PGD_EPSILONS, max_iter=PGD_MAX_ITER)
    df.insert(0, "seed", seed)
    df.insert(1, "model", "vqc")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / f"pgd_sweep_vqc_seed{seed}.csv", index=False)

    for _, row in df.iterrows():
        print(row.to_dict(), flush=True)

    return df


def run_pgd_sweep_for_seed_classical(seed: int) -> pd.DataFrame:
    X_sub, y_sub, clip_values = _load_scaled_subsample(seed)

    model = ClassicalBaseline(in_features=4, hidden_dim=8)
    model.load_state_dict(
        torch.load(MODELS_DIR / f"classical_baseline_seed{seed}.pt", weights_only=True)
    )
    model.eval()
    art_classifier = build_art_classifier(model, clip_values=clip_values, input_shape=(4,))

    df = run_pgd_sweep(art_classifier, X_sub, y_sub, PGD_EPSILONS, max_iter=PGD_MAX_ITER)
    df.insert(0, "seed", seed)
    df.insert(1, "model", "classical")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / f"pgd_sweep_classical_seed{seed}.csv", index=False)

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
        print(f"=== seed {seed} (vqc) ===", flush=True)
        all_dfs.append(run_pgd_sweep_for_seed_vqc(seed, base_config.circuit))
        print(f"=== seed {seed} (classical) ===", flush=True)
        all_dfs.append(run_pgd_sweep_for_seed_classical(seed))

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(SWEEP_DIR / "pgd_sweep_all_seeds.csv", index=False)
    print(f"\nWritten to {SWEEP_DIR / 'pgd_sweep_all_seeds.csv'}")


if __name__ == "__main__":
    main()
