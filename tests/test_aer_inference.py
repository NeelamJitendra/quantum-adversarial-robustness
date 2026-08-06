"""
Validates the shot-based Aer inference pipeline (src/noise/aer_inference.py)
against the exact StatevectorEstimator-based HybridClassifier it's meant
to approximate under noise=None -- catches qubit-indexing, parameter
binding, or classical-head-reconstruction bugs (the class of bug found
in the archived 06A notebook, where two cells silently disagreed on
which qubit the trained observable measured).

Requires results/models/vqc_seed42.pt (from the 8-seed sweep) to exist.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CircuitConfig
from src.models.hybrid_classifier import HybridClassifier
from src.models.quantum_model import create_model
from src.noise.aer_inference import evaluate_under_noise, load_checkpoint_params

CHECKPOINT_PATH = Path("results/models/vqc_seed42.pt")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="requires a trained checkpoint from the 8-seed sweep",
)


def _load_small_test_subsample(n=20):
    X_train_full = np.load("data/binary/X_train.npy")
    y_train_full = np.load("data/binary/y_train.npy")
    X_test = np.load("data/binary/X_test.npy")
    y_test = np.load("data/binary/y_test.npy")

    X_train, _, y_train, _ = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=42,
    )
    scaler = StandardScaler().fit(X_train)

    return scaler.transform(X_test)[:n], y_test[:n]


def test_ideal_aer_matches_exact_statevector_predictions():
    X_small, _ = _load_small_test_subsample(n=20)

    circuit_config = CircuitConfig(ansatz_reps=4, observable_mode="single_z")

    quantum_model = create_model(observable_mode="single_z", ansatz_reps=4, seed=42)
    model = HybridClassifier(quantum_model, quantum_output_dim=1)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        exact_logits = model(
            torch.tensor(X_small, dtype=torch.float32)
        ).numpy().ravel()

    quantum_weights, classifier_weight, classifier_bias = load_checkpoint_params(
        CHECKPOINT_PATH
    )

    aer_logits = evaluate_under_noise(
        quantum_weights, classifier_weight, classifier_bias,
        X_small, circuit_config, noise_model=None,
        shots=8192, seed_simulator=123,
    )

    # Sign (i.e. predicted class) must match exactly -- this is what
    # actually matters for accuracy/attack-success-rate downstream.
    assert np.array_equal(np.sign(exact_logits), np.sign(aer_logits))

    # Logit magnitudes should correlate strongly; small absolute
    # differences are expected shot noise at 8192 shots, not a bug.
    correlation = np.corrcoef(exact_logits, aer_logits)[0, 1]
    assert correlation > 0.99
