"""
Evaluation metrics for binary classification, and the adversarial /
noise-repeatability metrics named in the thesis proposal
(classification accuracy, attack success rate, robustness accuracy,
loss stability, confidence variance).
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    dict
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def attack_success_rate(y_true, y_pred_clean, y_pred_adv):
    """
    Fraction of originally-correct samples that an attack flips to an
    incorrect prediction. Restricting to originally-correct samples
    (rather than all samples) is the standard, ART-aligned definition
    -- flipping an already-wrong prediction isn't a meaningful "attack
    success". Named explicitly here since the term is used
    inconsistently across the QML adversarial-robustness literature.

    Returns 0.0 (not NaN) if no samples were originally correct.
    """

    y_true = np.asarray(y_true)
    y_pred_clean = np.asarray(y_pred_clean)
    y_pred_adv = np.asarray(y_pred_adv)

    originally_correct = y_pred_clean == y_true

    if not originally_correct.any():
        return 0.0

    flipped = y_pred_adv[originally_correct] != y_true[originally_correct]

    return float(flipped.mean())


def robustness_accuracy(y_true, y_pred_adv):
    """Accuracy under perturbation (noise and/or adversarial attack).
    Same computation as compute_metrics()["accuracy"] -- a distinctly
    named alias so results tables read directly in the proposal's own
    terminology instead of a generic "accuracy" column that could mean
    the clean or the perturbed condition."""

    return float(accuracy_score(y_true, y_pred_adv))


def loss_stability(losses):
    """
    Variance and coefficient of variation of a per-sample or per-run
    loss across *repeated* noisy evaluations of the same fixed
    input(s) (e.g. re-running the same test set through the same
    checkpoint under the same noise model with different Aer shot
    seeds). This is distinct from cross-seed variance (which reflects
    training variability, not inference-time noise repeatability) --
    callers must pass losses from repeated evaluations of one fixed
    trained model, not losses from different seeds/checkpoints.

    Parameters
    ----------
    losses : array-like
        Loss values from repeated evaluations of one fixed checkpoint
        under one fixed (stochastic) condition.

    Returns
    -------
    dict
        mean, variance, coefficient_of_variation (std / mean, NaN if
        mean is 0).
    """

    losses = np.asarray(losses, dtype=float)
    mean = losses.mean()
    variance = losses.var(ddof=1) if len(losses) > 1 else 0.0
    std = np.sqrt(variance)

    return {
        "mean": float(mean),
        "variance": float(variance),
        "coefficient_of_variation": float(std / mean) if mean != 0 else float("nan"),
    }


def confidence_variance(probs):
    """
    Variance of predicted-class probabilities (e.g. sigmoid outputs)
    across *repeated* noisy evaluations of the same fixed input(s) --
    same repeated-evaluation requirement as loss_stability(): pass
    probabilities from repeated runs of one fixed checkpoint under one
    fixed stochastic condition, not across different seeds.

    Parameters
    ----------
    probs : array-like
        Shape (n_repeats,) for a single sample, or (n_repeats, n_samples)
        for a batch -- variance is computed along the repeats axis
        (axis=0) either way.

    Returns
    -------
    float or np.ndarray
        Scalar for a single sample, per-sample array for a batch.
    """

    probs = np.asarray(probs, dtype=float)

    return probs.var(axis=0, ddof=1) if probs.shape[0] > 1 else np.zeros(probs.shape[1:])