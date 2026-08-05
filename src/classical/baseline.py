"""
Classical baseline for comparison against the VQC (RQ5).

Supersedes the classical baseline in the archived notebook 02
(trained on full 10-class raw-pixel MNIST, not the binary/PCA task
the VQC uses) -- this one trains on the identical binary 0-vs-1 /
4-PCA-feature data so the comparison is fair.
"""

import torch.nn as nn


class ClassicalBaseline(nn.Module):
    """
    Lightweight feedforward binary classifier on the 4 PCA features.

    Parameters
    ----------
    in_features : int
        Number of input features (4, matching the VQC's one-feature-
        per-qubit encoding).

    hidden_dim : int
        Width of the single hidden layer. Kept small ("lightweight",
        per the thesis proposal) so parameter count stays in the same
        order of magnitude as the VQC (22 trainable parameters:
        20 ansatz weights + 2 classical head).
    """

    def __init__(self, in_features: int = 4, hidden_dim: int = 8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)
