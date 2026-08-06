"""
PGD (Projected Gradient Descent) epsilon sweep via ART's
ProjectedGradientDescent, mirroring fgsm.py's structure exactly so
results are directly comparable. PGD is an iterative, stronger attack
than FGSM's single linearized step -- included as the proposal's
optional stretch goal ("PGD attacks may be included if time permits"),
to check whether the non-monotonic attack-success curve found with
FGSM (07_fgsm_attack.ipynb) is an artifact of FGSM's single-step
linearization or a deeper property of the decision boundary.
"""

from typing import List

import numpy as np
import pandas as pd
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from src.evaluation.metrics import attack_success_rate, compute_metrics, robustness_accuracy


def run_pgd_sweep(
    art_classifier: PyTorchClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epsilons: List[float],
    max_iter: int = 10,
    eps_step_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    Generate PGD adversarial examples at each epsilon (L-infinity norm)
    and evaluate. Same return schema as fgsm.run_fgsm_sweep for direct
    comparability.

    Parameters
    ----------
    max_iter : int
        Number of PGD iterations. 10 is a common, moderate default
        balancing attack strength against compute cost (each iteration
        costs about as much as one FGSM step).

    eps_step_fraction : float
        Per-iteration step size as a fraction of epsilon
        (eps_step = eps * eps_step_fraction). 0.25 is a standard
        choice (4 steps to reach the full epsilon budget, with the
        remaining max_iter-4 iterations refining within it).
    """

    clean_logits = art_classifier.predict(X)
    clean_preds = np.argmax(clean_logits, axis=1)

    rows = []

    for eps in epsilons:
        if eps == 0.0:
            X_adv = X.copy()
        else:
            attack = ProjectedGradientDescent(
                estimator=art_classifier,
                eps=eps,
                eps_step=eps * eps_step_fraction,
                max_iter=max_iter,
                norm=np.inf,
                verbose=False,
            )
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
