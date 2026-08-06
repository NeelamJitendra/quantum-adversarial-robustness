"""
Inference-time noise sweep: loads a trained VQC checkpoint (from
src/experiments/train_seed.py) and evaluates it under each noise
condition (src/noise/levels.py), without retraining -- per the thesis
architecture, the VQC is trained once noiselessly and noise/attacks
are applied only at inference time, keeping the compute budget
tractable given expensive parameter-shift training.

Unlike training (parameter-shift gradients, ~30-45 min/epoch),
shot-based noisy *inference* is fast (~6-9ms/sample, empirically
calibrated -- see notebook/06_noise_models.ipynb), so the full test
set (2956 samples) is used directly for every seed/condition, no
subsampling needed.

Conditions swept: ideal (L0), depolarizing/phase_damping/readout each
at L1-L4, and one combined (depolarizing+readout) condition at L2 --
14 conditions total, matching the thesis architecture's noise-level
design (L4 is the stretch/stress-test level).

CLI usage:
    python -m src.experiments.evaluate_noise_sweep --seeds 42,43,44,45,46,47,48,49
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CircuitConfig
from src.evaluation.metrics import compute_metrics
from src.noise.aer_inference import evaluate_under_noise, load_checkpoint_params
from src.noise.levels import NOISE_LEVELS
from src.noise.models import build_noise_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "binary"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
SWEEP_DIR = PROJECT_ROOT / "results" / "noise_sweep"

CONDITIONS = (
    [("none", "L0")]
    + [
        (noise_type, level)
        for noise_type in ["depolarizing", "phase_damping", "readout"]
        for level in ["L1", "L2", "L3", "L4"]
    ]
    + [("combined", "L2")]
)


def _load_test_split(seed: int):
    """Same 80/20 stratified split (by seed) used at training time --
    the scaler is refit on that seed's train split so the test set is
    transformed identically to how it was during training."""

    X_train_full = np.load(DATA_DIR / "X_train.npy")
    y_train_full = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    X_train, _, _, _ = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=seed,
    )
    scaler = StandardScaler().fit(X_train)

    return scaler.transform(X_test), y_test


def run_noise_sweep_for_seed(
    seed: int, circuit_config: CircuitConfig, shots: int = 1024
) -> pd.DataFrame:
    checkpoint_path = MODELS_DIR / f"vqc_seed{seed}.pt"
    quantum_weights, classifier_weight, classifier_bias = load_checkpoint_params(
        checkpoint_path
    )

    X_test_s, y_test = _load_test_split(seed)

    rows = []

    for noise_type, level_name in CONDITIONS:
        noise_model = build_noise_model(noise_type, NOISE_LEVELS[level_name])

        t0 = time.time()
        logits = evaluate_under_noise(
            quantum_weights, classifier_weight, classifier_bias,
            X_test_s, circuit_config, noise_model=noise_model,
            shots=shots, seed_simulator=seed,
        )
        elapsed = time.time() - t0

        predictions = (logits >= 0).astype(int)
        metrics = compute_metrics(y_test.ravel(), predictions)
        metrics.pop("confusion_matrix")

        row = {
            "seed": seed,
            "noise_type": noise_type,
            "level": level_name,
            "elapsed_s": elapsed,
            **metrics,
        }
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / f"noise_sweep_seed{seed}.csv", index=False)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--shots", type=int, default=1024)
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
        df = run_noise_sweep_for_seed(seed, base_config.circuit, shots=args.shots)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(SWEEP_DIR / "noise_sweep_all_seeds.csv", index=False)
    print(f"\nWritten to {SWEEP_DIR / 'noise_sweep_all_seeds.csv'}")


if __name__ == "__main__":
    main()
