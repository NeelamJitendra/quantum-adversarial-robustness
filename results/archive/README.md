# Archive

Files here are not reproducible by any current code path and are kept
for provenance only, not as inputs to any notebook or script.

- `notebook04_best_model.pt`, `notebook04_training_history.csv`: from
  an earlier, since-abandoned draft of "notebook 04" before it was
  split into 04A (raw) / 04B (scaled). `Trainer.fit()` never writes to
  disk itself, so these were saved manually by that earlier draft.
- `vqc_04b_seed42.pt` (lowercase b): an intermediate checkpoint from
  the same abandoned draft, superseded by `vqc_04B_scaled_seed42.pt`
  (uppercase B) in `results/models/`.
