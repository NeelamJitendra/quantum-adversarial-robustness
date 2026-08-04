"""
Regression test for two compounding non-determinism bugs found while
investigating why notebooks 04B and 05 each printed several different
"clean accuracy" numbers across cells for what should have been the
same trained model:

1. TorchConnector draws its default initial ansatz weights from
   PyTorch's *global* RNG (`self._weights.data.uniform_(-1, 1)`).
   `create_model(seed=...)` only ever passed `seed` to
   StatevectorEstimator, never to weight initialization -- so two
   "same seed" model instantiations actually started from genuinely
   different random weights (confirmed: max_abs_diff up to ~1.7 on a
   [-1, 1]-bounded output, i.e. not "estimator noise near a boundary"
   at all). Fixed in src/models/quantum_model.py:create_model() by
   drawing `initial_weights` from a local `np.random.Generator(seed)`.
2. With (1) fixed, StatevectorEstimator's exact (shot-free) simulation
   is confirmed bit-identical across instantiations for a fixed seed
   and input -- see the assertion below.

This compares the *quantum layer's* output directly (not wrapped in a
HybridClassifier), because HybridClassifier's classical nn.Linear head
is a second, separate source of unseeded random init and would
reintroduce the same class of confound if not pinned identically.
"""

import torch

from src.models.quantum_model import create_model
from src.utils.reproducibility import check_qnn_determinism


def _build_quantum_only(seed: int) -> torch.nn.Module:
    return create_model(observable_mode="single_z", seed=seed)


def test_qnn_output_determinism_same_seed():
    torch.manual_seed(0)
    x = torch.randn(16, 4)

    result = check_qnn_determinism(_build_quantum_only, x, seed=42)

    print(
        f"identical={result['identical']} "
        f"max_abs_diff={result['max_abs_diff']:.6f}"
    )

    assert result["identical"]
    assert result["max_abs_diff"] == 0.0
