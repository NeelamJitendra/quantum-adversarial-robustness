# Archive

Superseded notebooks, kept for reference. None of these are meant to
be re-run; see `../../MEMORY` plan notes / thesis architecture for the
current pipeline.

- `02_classical_baseline.ipynb`: trained on the wrong dataset (full
  10-class raw-pixel MNIST instead of the binary 0-vs-1 / 4-PCA-feature
  task) and gets 12% accuracy. Superseded by `05_classical_baseline`.
- `04A_clean_vqc_raw.ipynb`, `04B_clean_vqc_scaled.ipynb`: the
  pre-Phase-0-fix VQC training notebooks (~58% accuracy, see
  `04D_phase0_diagnostics`). Superseded by `04E_train_all_seeds`.
  Raw (unscaled) and scaled variants performed near-identically, so
  the raw path is dropped going forward — scaled-only is canonical.
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

`05_fgsm_clean_vqc_new.ipynb` (a 2-cell broken ART stub with an
unresolved NameError) was deleted rather than archived — nothing in it
was salvageable.
