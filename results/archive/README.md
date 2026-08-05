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
  together when `notebook/04C_quantum_output_analysis.ipynb` (the
  only thing that used them) was archived -- see
  `notebook/archive/README.md`. Superseded by the
  `single_z`/`ansatz_reps=4` sweep in `results/models/vqc_seed*.pt`.
