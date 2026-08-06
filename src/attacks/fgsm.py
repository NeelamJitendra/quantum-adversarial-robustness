"""
FGSM epsilon sweep via ART's FastGradientMethod, against an
ART-wrapped HybridClassifier (see art_wrapper.py).
"""

from typing import List

import numpy as np
import pandas as pd
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

from src.evaluation.metrics import attack_success_rate, compute_metrics, robustness_accuracy


def run_fgsm_sweep(
    art_classifier: PyTorchClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epsilons: List[float],
) -> pd.DataFrame:
    """
    Generate FGSM adversarial examples at each epsilon (L-infinity
    norm, standardized-feature units) and evaluate.

    Parameters
    ----------
    y : np.ndarray
        Integer class labels (0/1), matching ART's expected format
        (not the (-1, 1)-shaped float tensors used for BCE training).

    epsilons : list[float]
        Should include 0.0 as a sanity check: at eps=0.0,
        attack_success_rate must be 0 and robustness_accuracy must
        equal the clean accuracy.

    Returns
    -------
    pd.DataFrame
        One row per epsilon: accuracy/precision/recall/f1 (under
        attack), attack_success_rate, robustness_accuracy, and mean
        L-infinity / L2 perturbation magnitude.
    """

    clean_logits = art_classifier.predict(X)
    clean_preds = np.argmax(clean_logits, axis=1)

    rows = []

    for eps in epsilons:
        if eps == 0.0:
            X_adv = X.copy()
        else:
            attack = FastGradientMethod(estimator=art_classifier, eps=eps, norm=np.inf)
            X_adv = attack.generate(x=X)

        adv_logits = art_classifier.predict(X_adv)
        adv_preds = np.argmax(adv_logits, axis=1)

        metrics = compute_metrics(y, adv_preds)
        metrics.pop("confusion_matrix")

        perturbation = X_adv - X
        mean_linf = np.abs(perturbation).max(axis=1).mean()
        mean_l2 = np.linalg.norm(perturbation, axis=1).mean()

        rows.append({
            "epsilon": eps,
            "attack_success_rate": attack_success_rate(y, clean_preds, adv_preds),
            "robustness_accuracy": robustness_accuracy(y, adv_preds),
            "mean_linf": mean_linf,
            "mean_l2": mean_l2,
            **metrics,
        })

    return pd.DataFrame(rows)
