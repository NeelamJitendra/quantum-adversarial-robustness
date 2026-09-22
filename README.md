# Experimental Analysis of Adversarial Robustness in Variational Hybrid Quantum-Classical Machine Learning Models Under NISQ Noise

Master's thesis project. A 4-qubit variational quantum classifier (VQC) is
trained on binary MNIST (digits 0 vs. 1, reduced to 4 PCA features) and
compared against a classical feedforward baseline on the same task, then
both are evaluated under simulated NISQ noise, FGSM and PGD adversarial
attacks, and two inference-time mitigation techniques.

## Research questions

- **Main:** how does NISQ hardware noise affect the adversarial robustness
  of a hybrid quantum-classical model?
- **(a)** Which noise type has the greatest impact on robustness —
  depolarizing, phase damping, or readout error?
- **(b)** Does increasing noise intensity have a consistent effect on
  accuracy and on attack success rate?
- **(c)** Are accuracy and robustness under noise correlated?
- **(d)** Can readout calibration or zero-noise extrapolation improve
  robustness?
- **(e)** How does the hybrid model compare to an equivalent classical
  model, on both accuracy and robustness?

## Findings

- The VQC (`single_z` observable, `RealAmplitudes` depth 4) reaches 62.46%
  mean test accuracy across 8 seeds (std 0.85pt); the classical baseline on
  the identical features reaches 99.83% (std 0.09pt). A separate diagnostic
  pass (`notebook/04_phase0_diagnostics.ipynb`) ruled out gradient, learning
  rate, and mini-batch bugs before attributing the gap to the circuit's
  expressivity — this circuit family, without data re-uploading, cannot
  learn the full training set even though a plain logistic regression on
  the same 4 features gets 99.66%.
- Noise ranks depolarizing >> phase damping >> readout in impact on
  accuracy, formally significant after Benjamini-Hochberg correction
  (`notebook/06_noise_models.ipynb`, `notebook/10_statistical_analysis.ipynb`).
- FGSM against the VQC peaks at 58.4% attack success (epsilon=0.05) and is
  non-monotonic in epsilon; PGD reaches 95.7% at epsilon=0.1 — FGSM alone
  understates the model's vulnerability by close to 40 points
  (`notebook/07_fgsm_attack.ipynb`).
- Depolarizing noise mostly increases attack success as it increases,
  except at epsilon=0.05 — the model's single most vulnerable point —
  where more noise decreases it. Noise's effect on robustness depends on
  attack strength, not just noise level (`notebook/08_noise_attack_sweep.ipynb`).
- Readout calibration and zero-noise extrapolation, both validated correct
  in isolation, show no significant improvement in any of 7 tested
  conditions (all FDR-corrected p >= 0.73) — a genuine null result, with
  two candidate mechanisms discussed in the notebook
  (`notebook/09_mitigation.ipynb`).
- The classical baseline is not just more accurate — it is dramatically
  more adversarially robust, under both FGSM and PGD (Cohen's d beyond
  -9 in every comparison). Accuracy and robustness under noise are not
  significantly correlated (r=0.27, p=0.14) — the two axes behave partly
  independently (`notebook/10_statistical_analysis.ipynb`).

## Repository structure

```
notebook/       01-10, the pipeline in order (see below); archive/ holds
                 superseded notebooks with a README explaining why
src/             reusable pipeline code, organized by concern:
  circuits/      feature map and ansatz construction
  models/        the VQC, the classical baseline, the hybrid classifier
  noise/         Aer noise models and shot-based noisy inference
  attacks/       FGSM and PGD via the Adversarial Robustness Toolbox
  mitigation/    readout calibration and zero-noise extrapolation
  analysis/      paired significance testing (Wilcoxon, Cohen's d, FDR)
  experiments/   CLI entrypoints that run each sweep across seeds
  diagnostics/   the Phase 0 root-cause tooling
tests/           pytest suite for the above
results/         every sweep's output (CSVs, checkpoints, figures);
                 archive/ holds superseded artifacts with a README
data/            preprocessed MNIST (full and binary/PCA)
configs/         the experiment configuration used by every sweep
```

## Notebook sequence

| # | Notebook | Answers |
|---|---|---|
| 01 | Data preprocessing | — |
| 02 | Binary dataset preparation | — |
| 03 | Classical baseline | RQ(e) accuracy half |
| 04 | Phase 0 diagnostics | why the VQC underperforms |
| 05 | Train all seeds | VQC accuracy, 8 seeds |
| 06 | Noise models | RQ(a), RQ(b) accuracy trend |
| 07 | FGSM attack (+ PGD) | attack-strength comparison |
| 08 | Noise x attack sweep | main RQ, RQ(b) attack-success trend |
| 09 | Mitigation | RQ(d) |
| 10 | Statistical analysis | formal tests for every RQ above |

Each notebook is a thin wrapper: the actual computation lives in `src/` and
runs via the CLI entrypoints in `src/experiments/`, so a sweep can be
re-run without touching a notebook, e.g.:

```
python -m src.experiments.train_seed --seed 42
python -m src.experiments.run_all_seeds --seeds 42,43,44,45,46,47,48,49
python -m src.experiments.evaluate_noise_sweep --seeds 42,43,44,45,46,47,48,49
python -m src.experiments.evaluate_fgsm_sweep --seeds 42,43,44,45,46,47,48,49
python -m src.experiments.evaluate_pgd_sweep --seeds 42,43,44,45,46,47,48,49
python -m src.experiments.evaluate_mitigation_sweep --seeds 42,43,44,45,46,47,48,49
```

## Setup

Python 3.11, with:

```
python -m venv .venv
.venv\Scripts\Activate.ps1    # PowerShell; use .venv\Scripts\activate.bat for cmd.exe
pip install -r requirements.txt
```

Key dependencies: Qiskit 2.5.1, Qiskit Aer 0.17.2, Qiskit Machine Learning,
PyTorch, scikit-learn, Adversarial Robustness Toolbox.

## Tests

```
pytest tests/
```

39 tests covering observable/qubit-index mapping, weight-init
reproducibility, the noise and mitigation pipelines against exact
statevector simulation, the metrics module, and the statistics module.
