"""
Adapts HybridClassifier's single-logit binary output to the standard
multi-class logit API the Adversarial Robustness Toolbox (ART) expects.

ART's PyTorchClassifier is built around `nb_classes`-way classification
with CrossEntropyLoss over per-class logits. HybridClassifier outputs
a single scalar logit for a sigmoid/BCEWithLogitsLoss binary setup.
The standard, well-known trick for reusing multi-class tooling on a
binary sigmoid model: construct a 2-class logit vector
[-logit, logit] -- softmax over this pair reduces to the same decision
(argmax picks class 1 iff logit > 0, identical to the sigmoid
threshold at 0), and the transformation is differentiable end-to-end
through the same quantum layer (input_gradients=True is already
enabled on the underlying EstimatorQNN, required for FGSM's gradient
w.r.t. inputs).
"""

import torch
import torch.nn as nn


class BinaryLogitAdapter(nn.Module):
    """Wraps a HybridClassifier (single logit, shape (batch, 1)) as a
    2-class classifier (shape (batch, 2)) for ART."""

    def __init__(self, hybrid_model: nn.Module):
        super().__init__()
        self.hybrid_model = hybrid_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.hybrid_model(x)
        return torch.cat([-logit, logit], dim=1)
