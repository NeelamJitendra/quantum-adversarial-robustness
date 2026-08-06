"""
Tests for the adversarial/noise-repeatability metrics added to
src/evaluation/metrics.py (attack_success_rate, robustness_accuracy,
loss_stability, confidence_variance).
"""

import numpy as np

from src.evaluation.metrics import (
    attack_success_rate,
    confidence_variance,
    loss_stability,
    robustness_accuracy,
)


def test_attack_success_rate_only_counts_originally_correct_flips():
    y_true = np.array([0, 0, 1, 1])
    y_pred_clean = np.array([0, 1, 1, 0])  # sample 1, 3 already wrong
    y_pred_adv = np.array([1, 1, 0, 0])  # sample 0 flipped, sample 2 flipped

    # Originally correct: samples 0 and 2. Both flipped by the attack.
    assert attack_success_rate(y_true, y_pred_clean, y_pred_adv) == 1.0


def test_attack_success_rate_no_originally_correct_samples():
    y_true = np.array([0, 1])
    y_pred_clean = np.array([1, 0])  # both wrong already
    y_pred_adv = np.array([1, 0])

    assert attack_success_rate(y_true, y_pred_clean, y_pred_adv) == 0.0


def test_attack_success_rate_partial():
    y_true = np.array([0, 0, 0, 0])
    y_pred_clean = np.array([0, 0, 0, 0])  # all correct
    y_pred_adv = np.array([1, 1, 0, 0])  # half flipped

    assert attack_success_rate(y_true, y_pred_clean, y_pred_adv) == 0.5


def test_robustness_accuracy_matches_accuracy_score():
    y_true = np.array([0, 1, 1, 0])
    y_pred_adv = np.array([0, 1, 0, 0])

    assert robustness_accuracy(y_true, y_pred_adv) == 0.75


def test_loss_stability_constant_losses_have_zero_variance():
    result = loss_stability([0.5, 0.5, 0.5])

    assert result["mean"] == 0.5
    assert result["variance"] == 0.0
    assert result["coefficient_of_variation"] == 0.0


def test_loss_stability_single_value_has_zero_variance():
    result = loss_stability([0.7])

    assert result["variance"] == 0.0


def test_confidence_variance_single_sample():
    # 5 repeated runs of one sample -> scalar-shaped result
    probs = np.array([0.5, 0.6, 0.4, 0.5, 0.5])
    result = confidence_variance(probs)

    assert result == np.var(probs, ddof=1)


def test_confidence_variance_batch():
    # 3 repeats x 2 samples -> per-sample variance
    probs = np.array([
        [0.5, 0.9],
        [0.6, 0.9],
        [0.4, 0.9],
    ])
    result = confidence_variance(probs)

    assert result.shape == (2,)
    assert result[1] == 0.0  # constant across repeats
    assert result[0] > 0.0
