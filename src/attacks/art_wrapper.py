"""
Builds an ART PyTorchClassifier wrapping a trained HybridClassifier
via BinaryLogitAdapter.
"""

from typing import Tuple

import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

from src.attacks.art_adapter import BinaryLogitAdapter


def build_art_classifier(
    hybrid_model: nn.Module,
    clip_values: Tuple[float, float],
    input_shape: Tuple[int, ...] = (4,),
) -> PyTorchClassifier:
    """
    Parameters
    ----------
    hybrid_model : nn.Module
        A trained HybridClassifier (single logit output). Set to
        eval() by the caller before passing in, if not already.

    clip_values : (float, float)
        (min, max) of the feature space attacks are allowed to
        perturb within -- pass the actual observed
        StandardScaler-transformed training data range, not an
        arbitrary bound.

    input_shape : tuple
        Per-sample feature shape (4 PCA features, one per qubit).
    """

    adapter = BinaryLogitAdapter(hybrid_model)
    adapter.eval()

    return PyTorchClassifier(
        model=adapter,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=2,
        clip_values=clip_values,
    )
