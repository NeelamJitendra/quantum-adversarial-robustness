"""
Hybrid quantum-classical classifier.

The quantum model is injected into the classifier so that
different quantum configurations can be tested independently.
"""

import torch
import torch.nn as nn


class HybridClassifier(nn.Module):
    """
    Hybrid quantum-classical binary classifier.

    Parameters
    ----------
    quantum_model : nn.Module
        Quantum neural network wrapped as a PyTorch module.
    """

    def __init__(
        self,
        quantum_model,
    ):
        super().__init__()

        self.quantum = quantum_model

        self.classifier = nn.Linear(
            in_features=1,
            out_features=1,
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Binary classification logits.
        """

        x = self.quantum(x)

        x = self.classifier(x)

        return x