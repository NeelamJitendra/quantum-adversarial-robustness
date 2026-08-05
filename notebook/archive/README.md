# Archive

Superseded notebooks, kept for reference. None of these are meant to
be re-run; see `../../MEMORY` plan notes / thesis architecture for the
current pipeline.

- `02_classical_baseline.ipynb`: trained on the wrong dataset (full
  10-class raw-pixel MNIST instead of the binary 0-vs-1 / 4-PCA-feature
  task) and gets 12% accuracy. Superseded by `03_classical_baseline`.
- `04A_clean_vqc_raw.ipynb`, `04B_clean_vqc_scaled.ipynb`: the
  pre-Phase-0-fix VQC training notebooks (~58% accuracy, see
  `04_phase0_diagnostics`). Superseded by `05_train_all_seeds`.
  Raw (unscaled) and scaled variants performed near-identically, so
  the raw path is dropped going forward — scaled-only is canonical.
- `04C_quantum_output_analysis.ipynb`: quantum-output/effect-size
  analysis (Cohen's d, class separation), but built on the
  pre-Phase-0-fix `vqc_04B_scaled_seed42.pt` checkpoint (~58%
  accuracy) -- stale now that `single_z`/`ansatz_reps=4` is the
  accepted architecture. Its dependent artifacts
  (`vqc_04A_raw_seed42.pt`, `vqc_04B_scaled_seed42.pt`+metadata,
  `standard_scaler_04B_seed42.joblib`,
  `04C_quantum_output_analysis_seed42.json`) moved to
  `results/archive/` alongside it. The analysis approach itself is
  reusable -- worth rebuilding against a post-sweep checkpoint later,
  not yet superseded by a numbered notebook.
- `05_fgsm_clean_vqc.ipynb`: working hand-rolled FGSM (not ART).
  Methodologically fine, but the thesis proposal specifies ART
  specifically. Its plotting/reporting code is reused in the new
  `07_fgsm_attack` / `10_statistical_analysis` notebooks. Superseded
  for the attack implementation itself.
- `06A_build_noise_model.ipynb`: real depolarizing/readout NoiseModel
  construction and shot-based noisy inference, correct Aer API usage,
  but has an unresolved qubit-indexing bug (two cells disagree on
  which qubit index the trained "ZIII" observable corresponds to) and
  never persisted a results table. Core NoiseModel/AerSimulator logic
  is ported into `src/noise/`; superseded by `06_noise_models`.

**Renumbering note:** the archived `03_MNIST_0_vs_1.ipynb` name doesn't
appear above because it wasn't archived -- it was renamed (not
superseded) to `02_binary_dataset_preparation.ipynb`, since `02` was a
gap left by archiving the old classical baseline. Similarly
`04D_phase0_diagnostics.ipynb` was renamed to `04_phase0_diagnostics.ipynb`
(no remaining `04A`/`04B`/`04C` in the active folder to distinguish it
from).

`05_fgsm_clean_vqc_new.ipynb` (a 2-cell broken ART stub with an
unresolved NameError) was deleted rather than archived — nothing in it
was salvageable.
