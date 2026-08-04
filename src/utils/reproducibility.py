"""
Determinism check for StatevectorEstimator-backed quantum models.

Notebooks 04B and 05 both independently discovered that two separate
instantiations of the same quantum model, given the same seed and the
same input, do not always produce bit-identical outputs (small
differences accumulate near the decision boundary). This module makes
that check explicit and repeatable instead of leaving it as an ad hoc
notebook investigation.
"""

from typing import Callable

import numpy as np
import torch


def check_qnn_determinism(
    model_factory: Callable[[int], torch.nn.Module],
    x: torch.Tensor,
    seed: int = 42,
    atol: float = 0.0,
) -> dict:
    """
    Build the same model twice from `model_factory(seed)` and compare
    outputs on the same fixed input.

    Parameters
    ----------
    model_factory : Callable[[int], torch.nn.Module]
        Factory that builds a fresh model given a seed, e.g.
        `lambda seed: HybridClassifier(create_model(seed=seed))`.

    x : torch.Tensor
        Fixed input batch to evaluate both instantiations on.

    seed : int
        Seed passed to both instantiations.

    atol : float
        Absolute tolerance for the identity check. 0.0 requires
        bit-identical output.

    Returns
    -------
    dict
        identical : bool
        max_abs_diff : float
        out1, out2 : torch.Tensor
    """

    model1 = model_factory(seed)
    model2 = model_factory(seed)

    model1.eval()
    model2.eval()

    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    max_abs_diff = (out1 - out2).abs().max().item()
    identical = np.allclose(
        out1.detach().numpy(),
        out2.detach().numpy(),
        atol=atol,
    )

    return {
        "identical": identical,
        "max_abs_diff": max_abs_diff,
        "out1": out1,
        "out2": out2,
    }
