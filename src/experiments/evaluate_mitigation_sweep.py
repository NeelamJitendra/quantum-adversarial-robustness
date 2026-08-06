"""
Mitigation evaluation pipeline: applies readout calibration and ZNE
(src/mitigation/) on top of the trained checkpoints under selected
noise conditions, at inference time, and compares against the
unmitigated results already computed in evaluate_noise_sweep.py.

Scope (per the thesis architecture -- mitigation is the most droppable
item, kept intentionally narrow):
- Readout calibration -> applied to readout-L2, readout-L4, and
  combined-L2 (the conditions it specifically targets -- readout
  error, not gate error). One calibration matrix per noise condition
  is shared across all 8 seeds (it depends only on the noise model,
  not the trained weights).
- ZNE -> applied to depolarizing-L2, depolarizing-L4, phase_damping-L2,
  phase_damping-L4 (gate-noise conditions; readout error isn't a gate
  error and folding doesn't target it).

Runs on a class-stratified 300-sample subsample (same convention as
evaluate_fgsm_sweep.py / evaluate_noise_attack_sweep.py, fixed across
seeds), per the thesis architecture's ZNE compute-budget guidance
(~200-300 points, not the full test set, given the ~3x shot cost of
folding at 3 scale factors).

CLI usage:
    python -m src.experiments.evaluate_mitigation_sweep --seeds 42,43,44,45,46,47,48,49
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_metrics
from src.experiments.evaluate_fgsm_sweep import select_subsample
from src.mitigation.readout_calibration import build_calibration_matrix, predict_with_calibration
from src.mitigation.zne import zne_predict
from src.noise.aer_inference import load_checkpoint_params
from src.noise.levels import NOISE_LEVELS
from src.noise.models import build_noise_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "binary"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
SWEEP_DIR = PROJECT_ROOT / "results" / "mitigation_sweep"

READOUT_CALIBRATION_CONDITIONS = [
    ("readout", "L2"),
    ("readout", "L4"),
    ("combined", "L2"),
]
ZNE_CONDITIONS = [
    ("depolarizing", "L2"),
    ("depolarizing", "L4"),
    ("phase_damping", "L2"),
    ("phase_damping", "L4"),
]

SUBSAMPLE_SIZE = 300
SUBSAMPLE_SEED = 123  # same convention as evaluate_fgsm_sweep.py


def run_mitigation_sweep_for_seed(seed: int, circuit_config, shots: int = 1024) -> pd.DataFrame:
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

    X_sub, y_sub = select_subsample(X_test_s, y_test, n=SUBSAMPLE_SIZE)
    X_sub = X_sub.astype(np.float32)
    y_sub = y_sub.astype(np.int64)

    quantum_weights, classifier_weight, classifier_bias = load_checkpoint_params(
        MODELS_DIR / f"vqc_seed{seed}.pt"
    )

    rows = []

    for noise_type, level_name in READOUT_CALIBRATION_CONDITIONS:
        noise_model = build_noise_model(noise_type, NOISE_LEVELS[level_name])
        cal_matrix = build_calibration_matrix(
            noise_model, num_qubits=circuit_config.num_qubits,
            shots=4096, seed_simulator=42,  # fixed, independent of model seed -- shared matrix
        )
        logits = predict_with_calibration(
            quantum_weights, classifier_weight, classifier_bias,
            X_sub, circuit_config, noise_model, cal_matrix,
            shots=shots, seed_simulator=seed,
        )
        predictions = (logits >= 0).astype(int)
        metrics = compute_metrics(y_sub, predictions)
        metrics.pop("confusion_matrix")

        row = {
            "seed": seed, "method": "readout_calibration",
            "noise_type": noise_type, "level": level_name,
            **metrics,
        }
        rows.append(row)
        print(row, flush=True)

    for noise_type, level_name in ZNE_CONDITIONS:
        noise_model = build_noise_model(noise_type, NOISE_LEVELS[level_name])
        logits = zne_predict(
            quantum_weights, classifier_weight, classifier_bias,
            X_sub, circuit_config, noise_model,
            scale_factors=(1, 3, 5), shots=shots, seed_simulator=seed,
        )
        predictions = (logits >= 0).astype(int)
        metrics = compute_metrics(y_sub, predictions)
        metrics.pop("confusion_matrix")

        row = {
            "seed": seed, "method": "zne",
            "noise_type": noise_type, "level": level_name,
            **metrics,
        }
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / f"mitigation_sweep_seed{seed}.csv", index=False)

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
        all_dfs.append(run_mitigation_sweep_for_seed(seed, base_config.circuit, shots=args.shots))

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(SWEEP_DIR / "mitigation_sweep_all_seeds.csv", index=False)
    print(f"\nWritten to {SWEEP_DIR / 'mitigation_sweep_all_seeds.csv'}")


if __name__ == "__main__":
    main()
