"""
Centralized seeding for reproducibility.

Every notebook up to this point (04A, 04B, 05) defined its own local
set_seed() closure. This is the single implementation all of them
should import instead.
"""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed all RNGs used across the project (Python, NumPy, PyTorch).

    Parameters
    ----------
    seed : int
        Seed value.

    deterministic : bool
        If True, also request deterministic PyTorch algorithms where
        available. Quantum circuit evaluation (StatevectorEstimator)
        is seeded separately wherever the estimator is constructed
        (see src/models/quantum_model.py) since it is not covered by
        torch's RNG.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
