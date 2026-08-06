# Archive

Files here are not reproducible by any current code path and are kept
for provenance only, not as inputs to any notebook or script.

- `notebook04_best_model.pt`, `notebook04_training_history.csv`: from
  an earlier, since-abandoned draft of "notebook 04" before it was
  split into 04A (raw) / 04B (scaled). `Trainer.fit()` never writes to
  disk itself, so these were saved manually by that earlier draft.
- `vqc_04b_seed42.pt` (lowercase b): an intermediate checkpoint from
  the same abandoned draft, superseded by `vqc_04B_scaled_seed42.pt`
  (uppercase B), also archived here now (see below).
- `vqc_04A_raw_seed42.pt`, `vqc_04B_scaled_seed42.pt`+metadata,
  `standard_scaler_04B_seed42.joblib`,
  `04C_quantum_output_analysis_seed42.json`: pre-Phase-0-fix
  artifacts (~58% accuracy, `single_z`/`ansatz_reps=2`). Moved here
  together when `notebook/archive/quantum_output_analysis_prephase0fix.ipynb`
  (the only thing that used them) was archived -- see
  `notebook/archive/README.md`. Superseded by the
  `single_z`/`ansatz_reps=4` sweep in `results/models/vqc_seed*.pt`.
- `clean_predictions_04B_seed42.npy`, `clean_reference_04B_seed42.json`,
  `fgsm_04B_seed42.{csv,json}`, `fgsm_*.png` (5 plots): FGSM results
  from the same pre-Phase-0-fix `vqc_04B_scaled_seed42.pt` checkpoint,
  produced by the archived `notebook/archive/fgsm_handrolled_prephase0fix.ipynb`.
  Moved from the now-empty `results/adversarial/`. Superseded once
  `07_fgsm_attack` (ART-based, per the thesis proposal) ran against a
  current checkpoint.
- `experiment_06A_config.json`: noise-model hyperparameter config from
  `notebook/archive/noise_model_prototype.ipynb` -- missed in the
  first archival pass, moved here from the now-empty `results/noise/`.
  Superseded by `src/noise/levels.py`'s `NOISE_LEVELS` table.
