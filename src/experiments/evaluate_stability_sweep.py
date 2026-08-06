"""
Measures loss_stability and confidence_variance (src/evaluation/metrics.py)
-- two metrics named explicitly in the thesis proposal's Section 4.6
("classification accuracy, attack success rate, robustness accuracy,
loss stability, and confidence variance") that were implemented and
tested but never actually computed anywhere.

Both require *repeated* evaluations of the same fixed checkpoint on
the same fixed inputs under the same (stochastic) noise condition,
varying only the Aer shot-sampling seed -- this isolates shot-noise-
induced instability from cross-seed training variance, which is a
different, already-covered axis (results/models/vqc_all_seeds_summary.csv
etc.).

Representative conditions: depolarizing L2 (device-realistic) and L4
(stress-test) -- the two levels already used in the mitigation sweep,
for consistency -- on a 100-sample subsample, 10 repeats per
(seed, level, sample).

CLI usage:
    python -m src.experiments.evaluate_stability_sweep --seeds 42,43,44,45,46,47,48,49
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import confidence_variance, loss_stability
from src.experiments.evaluate_fgsm_sweep import select_subsample
from src.noise.aer_inference import evaluate_under_noise, load_checkpoint_params
from src.noise.levels import NOISE_LEVELS
from src.noise.models import build_noise_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "binary"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
SWEEP_DIR = PROJECT_ROOT / "results" / "stability_sweep"

LEVELS = ["L2", "L4"]
SUBSAMPLE_N = 100
N_REPEATS = 10


def run_stability_sweep_for_seed(seed: int, circuit_config) -> pd.DataFrame:
    X_train_full = np.load(DATA_DIR / "X_train.npy")
    y_train_full = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    X_train, _, _, _ = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=seed,
    )
    scaler = StandardScaler().fit(X_train)
    X_test_s = scaler.transform(X_test)

    X_sub, y_sub = select_subsample(X_test_s, y_test, n=SUBSAMPLE_N)
    X_sub = X_sub.astype(np.float32)
    y_sub = torch.tensor(y_sub, dtype=torch.float32).reshape(-1, 1)

    quantum_weights, classifier_weight, classifier_bias = load_checkpoint_params(
        MODELS_DIR / f"vqc_seed{seed}.pt"
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    rows = []

    for level in LEVELS:
        noise_model = build_noise_model("depolarizing", NOISE_LEVELS[level])

        # (n_repeats, n_samples) arrays -- one row per repeat, one
        # column per sample, varying only the Aer shot-sampling seed.
        all_logits = np.empty((N_REPEATS, len(X_sub)))
        for repeat in range(N_REPEATS):
            all_logits[repeat] = evaluate_under_noise(
                quantum_weights, classifier_weight, classifier_bias,
                X_sub, circuit_config, noise_model=noise_model,
                shots=1024, seed_simulator=seed * 1000 + repeat,
            )

        logits_t = torch.tensor(all_logits, dtype=torch.float32)
        y_broadcast = y_sub.T.expand(N_REPEATS, -1)
        losses = criterion(logits_t, y_broadcast).numpy()  # (n_repeats, n_samples)
        probs = torch.sigmoid(logits_t).numpy()  # (n_repeats, n_samples)

        per_sample_conf_var = confidence_variance(probs)  # (n_samples,)
        per_sample_loss_stats = [loss_stability(losses[:, i]) for i in range(losses.shape[1])]
        per_sample_loss_cv = np.array([s["coefficient_of_variation"] for s in per_sample_loss_stats])

        row = {
            "seed": seed,
            "noise_type": "depolarizing",
            "level": level,
            "n_repeats": N_REPEATS,
            "n_samples": len(X_sub),
            "mean_confidence_variance": float(np.mean(per_sample_conf_var)),
            "mean_loss_coefficient_of_variation": float(np.nanmean(per_sample_loss_cv)),
            "mean_loss": float(losses.mean()),
        }
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / f"stability_sweep_seed{seed}.csv", index=False)

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
    all_dfs = [run_stability_sweep_for_seed(seed, base_config.circuit) for seed in seeds]

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(SWEEP_DIR / "stability_sweep_all_seeds.csv", index=False)
    print(f"\nWritten to {SWEEP_DIR / 'stability_sweep_all_seeds.csv'}")


if __name__ == "__main__":
    main()
