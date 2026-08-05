"""
Aggregates per-seed VQC training results into one summary table.

Each seed is trained independently (see train_seed.py's CLI) --
typically launched as separate parallel OS processes for wall-clock
efficiency (parallel scaling was empirically measured before the
first real sweep: ~2.4x net throughput at 4 concurrent processes,
~3.3x at 8, on a 16-logical-core machine -- see
notebook/04D_phase0_diagnostics.ipynb / the thesis architecture log
for the calibration). This script does not itself train; it just
collects results/models/vqc_seed{N}_metadata.json for each requested
seed into results/models/vqc_all_seeds_summary.csv.

CLI usage (after training each seed separately):
    python -m src.experiments.run_all_seeds --seeds 42,43,44,45,46,47,48,49
"""

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "results" / "models"


def collect_summary(seeds: list) -> pd.DataFrame:
    rows = []

    for seed in seeds:
        metadata_path = MODELS_DIR / f"vqc_seed{seed}_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No metadata for seed {seed} at {metadata_path} -- "
                f"train it first (python -m src.experiments.train_seed --seed {seed})"
            )

        metadata = json.loads(metadata_path.read_text())
        metrics = metadata["test_metrics"]

        rows.append({
            "seed": seed,
            "best_val_loss": metadata["best_val_loss"],
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    df = collect_summary(seeds)

    out_path = MODELS_DIR / "vqc_all_seeds_summary.csv"
    df.to_csv(out_path, index=False)

    print(df)
    print()
    print(df[["test_accuracy", "test_precision", "test_recall", "test_f1"]].agg(["mean", "std"]))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
