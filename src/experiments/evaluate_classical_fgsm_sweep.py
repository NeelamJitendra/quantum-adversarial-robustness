"""
FGSM sweep against the classical baseline -- closes the gap where
RQ(e) ("how does the robustness of noisy hybrid quantum-classical
models compare with equivalent classical machine learning models?")
was only ever evaluated on clean accuracy (05_train_all_seeds,
10_statistical_analysis), never on adversarial robustness. Mirrors
evaluate_fgsm_sweep.py exactly (same epsilon grid, same class-
stratified 500-sample subsample convention, same ART BinaryLogitAdapter
trick -- ClassicalBaseline outputs a single logit of shape (batch, 1),
identical to HybridClassifier, so the same adapter applies with no
modification) so the two attack curves are directly comparable.

CLI usage:
    python -m src.experiments.evaluate_classical_fgsm_sweep --seeds 42,43,44,45,46,47,48,49
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
from src.classical.baseline import ClassicalBaseline
from src.experiments.evaluate_fgsm_sweep import EPSILONS, select_subsample

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "binary"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
SWEEP_DIR = PROJECT_ROOT / "results" / "classical_fgsm_sweep"


def run_classical_fgsm_sweep_for_seed(seed: int) -> pd.DataFrame:
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

    model = ClassicalBaseline(in_features=4, hidden_dim=8)
    model.load_state_dict(
        torch.load(MODELS_DIR / f"classical_baseline_seed{seed}.pt", weights_only=True)
    )
    model.eval()

    clip_values = (float(X_train_s.min()), float(X_train_s.max()))
    art_classifier = build_art_classifier(model, clip_values=clip_values, input_shape=(4,))

    df = run_fgsm_sweep(art_classifier, X_sub, y_sub, EPSILONS)
    df.insert(0, "seed", seed)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / f"classical_fgsm_sweep_seed{seed}.csv", index=False)

    for _, row in df.iterrows():
        print(row.to_dict(), flush=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    all_dfs = [run_classical_fgsm_sweep_for_seed(seed) for seed in seeds]

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(SWEEP_DIR / "classical_fgsm_sweep_all_seeds.csv", index=False)
    print(f"\nWritten to {SWEEP_DIR / 'classical_fgsm_sweep_all_seeds.csv'}")


if __name__ == "__main__":
    main()
