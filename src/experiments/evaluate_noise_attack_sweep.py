"""
Noise x attack composition sweep -- the pipeline that actually answers
the thesis's main RQ and sub-RQ(a)/(b) for adversarial robustness
specifically (noise-only and attack-only sweeps each answer a
narrower question in isolation).

Generates FGSM adversarial examples against the clean (noiseless)
model's gradient once per (seed, epsilon) -- the "deployable,
attacker-knowable" model -- then evaluates those *same fixed*
adversarial examples under each noise condition at inference time.

Methodological note (per the thesis architecture plan): adversarial
examples are NOT generated against a noise-affected model directly.
ART's FGSM needs a differentiable, deterministic forward pass; a
shot-based Aer forward pass under noise is stochastic and not
straightforwardly differentiable through the parameter-shift chain,
so an adaptive noise-aware attacker would be a much larger engineering
lift than this thesis's timeline supports. This measures attack
*transfer* under hardware noise, not an adaptive attacker -- a
documented scope limitation, not an oversight (matches the framing of
Winderl et al. 2024 and West et al. 2023, already in the proposal's
bibliography).

Includes the eps=0.0 x every-noise-condition and every-epsilon x
none/L0 cells even though they duplicate evaluate_noise_sweep.py and
evaluate_fgsm_sweep.py results respectively -- the redundancy is cheap
(noise evaluation is fast) and doubles as a cross-pipeline consistency
check.

CLI usage:
    python -m src.experiments.evaluate_noise_attack_sweep --seeds 42,43,44,45,46,47,48,49
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from art.attacks.evasion import FastGradientMethod

from src.attacks.art_wrapper import build_art_classifier
from src.evaluation.metrics import attack_success_rate, compute_metrics, robustness_accuracy
from src.experiments.evaluate_fgsm_sweep import EPSILONS, load_model, select_subsample
from src.noise.aer_inference import evaluate_under_noise, load_checkpoint_params
from src.noise.levels import NOISE_LEVELS
from src.noise.models import build_noise_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "binary"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
SWEEP_DIR = PROJECT_ROOT / "results" / "noise_attack_sweep"

NOISE_CONDITIONS = (
    [("none", "L0")]
    + [
        (noise_type, level)
        for noise_type in ["depolarizing", "phase_damping", "readout"]
        for level in ["L1", "L2", "L3", "L4"]
    ]
    + [("combined", "L2")]
)


def run_noise_attack_sweep_for_seed(seed: int, circuit_config, shots: int = 1024) -> pd.DataFrame:
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

    quantum_weights, classifier_weight, classifier_bias = load_checkpoint_params(
        MODELS_DIR / f"vqc_seed{seed}.pt"
    )

    # Clean (noiseless) predictions -- the "originally correct" reference
    # for attack_success_rate, matching evaluate_fgsm_sweep.py's convention.
    clean_preds = np.argmax(art_classifier.predict(X_sub), axis=1)

    rows = []

    for eps in EPSILONS:
        if eps == 0.0:
            X_adv = X_sub.copy()
        else:
            attack = FastGradientMethod(estimator=art_classifier, eps=eps, norm=np.inf)
            X_adv = attack.generate(x=X_sub)

        for noise_type, level_name in NOISE_CONDITIONS:
            noise_model = build_noise_model(noise_type, NOISE_LEVELS[level_name])

            logits = evaluate_under_noise(
                quantum_weights, classifier_weight, classifier_bias,
                X_adv, circuit_config, noise_model=noise_model,
                shots=shots, seed_simulator=seed,
            )
            predictions = (logits >= 0).astype(int)

            metrics = compute_metrics(y_sub, predictions)
            metrics.pop("confusion_matrix")

            row = {
                "seed": seed,
                "epsilon": eps,
                "noise_type": noise_type,
                "level": level_name,
                "attack_success_rate": attack_success_rate(y_sub, clean_preds, predictions),
                "robustness_accuracy": robustness_accuracy(y_sub, predictions),
                **metrics,
            }
            rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(rows)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / f"noise_attack_sweep_seed{seed}.csv", index=False)

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
        all_dfs.append(
            run_noise_attack_sweep_for_seed(seed, base_config.circuit, shots=args.shots)
        )

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(SWEEP_DIR / "noise_attack_sweep_all_seeds.csv", index=False)
    print(f"\nWritten to {SWEEP_DIR / 'noise_attack_sweep_all_seeds.csv'}")


if __name__ == "__main__":
    main()
