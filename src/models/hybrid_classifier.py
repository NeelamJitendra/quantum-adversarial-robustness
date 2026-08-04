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

    quantum_output_dim : int
        Width of quantum_model's output (1 for the legacy single-Z
        observable, num_qubits for the multi-Z observable -- see
        src/models/quantum_model.py:get_output_dim()). Defaults to 1
        to preserve the original single-output behaviour.
    """

    def __init__(
        self,
        quantum_model,
        quantum_output_dim: int = 1,
    ):
        super().__init__()

        self.quantum = quantum_model

        self.classifier = nn.Linear(
            in_features=quantum_output_dim,
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