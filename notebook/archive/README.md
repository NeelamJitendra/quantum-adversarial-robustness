# Archive

Superseded notebooks, kept for reference. None of these are meant to
be re-run; see the thesis architecture plan for the current pipeline.

Files were originally numbered to match whatever the active sequence
was at the time they were superseded (`02_classical_baseline.ipynb`,
`04A_clean_vqc_raw.ipynb`, etc.) -- since the active sequence has since
been renumbered (`01`-`10`), those prefixes now collide with unrelated
current notebooks and were confusing. Renamed to plain descriptive
names, no numeric prefix, so nothing here is mistakable for part of
the active pipeline.

- `classical_baseline_10class_broken.ipynb` (was `02_classical_baseline.ipynb`):
  trained on the wrong dataset (full 10-class raw-pixel MNIST instead
  of the binary 0-vs-1 / 4-PCA-feature task) and gets 12% accuracy.
  Superseded by `03_classical_baseline`.
- `vqc_raw_features_prephase0fix.ipynb`, `vqc_scaled_features_prephase0fix.ipynb`
  (were `04A_clean_vqc_raw.ipynb`, `04B_clean_vqc_scaled.ipynb`): the
  pre-Phase-0-fix VQC training notebooks (~58% accuracy, see
  `04_phase0_diagnostics`). Superseded by `05_train_all_seeds`. Raw
  (unscaled) and scaled variants performed near-identically, so the
  raw path is dropped going forward -- scaled-only is canonical.
- `quantum_output_analysis_prephase0fix.ipynb` (was `04C_quantum_output_analysis.ipynb`):
  quantum-output/effect-size analysis (Cohen's d, class separation),
  but built on the pre-Phase-0-fix `vqc_04B_scaled_seed42.pt`
  checkpoint (~58% accuracy) -- stale now that `single_z`/
  `ansatz_reps=4` is the accepted architecture. Its dependent
  artifacts (`vqc_04A_raw_seed42.pt`, `vqc_04B_scaled_seed42.pt`+metadata,
  `standard_scaler_04B_seed42.joblib`,
  `04C_quantum_output_analysis_seed42.json`) moved to
  `results/archive/` alongside it. The analysis approach itself is
  reusable -- worth rebuilding against a post-sweep checkpoint later,
  not yet superseded by a numbered notebook. Note: this file is
  referenced by path from `thesis/thesis_draft.md` (the paired-Cohen's-d
  convention citation) -- that reference was NOT auto-updated when this
  file was renamed, since `thesis/` is actively being written in a
  separate session; update it there if needed.
- `fgsm_handrolled_prephase0fix.ipynb` (was `05_fgsm_clean_vqc.ipynb`):
  working hand-rolled FGSM (not ART). Methodologically fine, but the
  thesis proposal specifies ART specifically. Its plotting/reporting
  code is reused in the new `07_fgsm_attack` / `10_statistical_analysis`
  notebooks. Superseded for the attack implementation itself. Its
  result artifacts (all of the former `results/adversarial/`:
  `clean_predictions_04B_seed42.npy`, `clean_reference_04B_seed42.json`,
  `fgsm_04B_seed42.{csv,json}`, and 5 `fgsm_*.png` plots) were computed
  against the same pre-Phase-0-fix `vqc_04B_scaled_seed42.pt`
  checkpoint, so they moved to `results/archive/` too.
- `noise_model_prototype.ipynb` (was `06A_build_noise_model.ipynb`):
  real depolarizing/readout NoiseModel construction and shot-based
  noisy inference, correct Aer API usage, but has an unresolved
  qubit-indexing bug (two cells disagree on which qubit index the
  trained "ZIII" observable corresponds to) and never persisted a
  results table. Core NoiseModel/AerSimulator logic is ported into
  `src/noise/`; superseded by `06_noise_models`. Its config
  (`experiment_06A_config.json`) moved to `results/archive/`.
- `00_qiskit_basics_scratch.ipynb`, `00_noise_model_scratch.ipynb`
  (were `notebook/old/01_qiskit_basics.ipynb`,
  `notebook/old/02_noise_models.ipynb`): pre-project Qiskit learning
  exploration (single-qubit Hadamard/measure, single-qubit
  depolarizing noise) -- self-archived by the student before this
  project's pipeline existed. Consolidated here from the old
  `notebook/old/` directory (now removed) so there's one archive
  location, not two. `00` prefix marks them as predating the real
  numbered sequence, not a gap within it.

`05_fgsm_clean_vqc_new.ipynb` (a 2-cell broken ART stub with an
unresolved NameError) was deleted rather than archived — nothing in it
was salvageable.
