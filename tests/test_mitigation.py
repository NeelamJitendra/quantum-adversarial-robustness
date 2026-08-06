"""
Tests for src/mitigation/ -- readout calibration and ZNE.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from src.circuits.ansatz import create_ansatz
from src.circuits.feature_map import create_feature_map
from src.mitigation.readout_calibration import (
    apply_readout_mitigation,
    build_calibration_matrix,
)
from src.mitigation.zne import fold_circuit, richardson_extrapolate
from src.noise.models import build_readout_noise_model


def test_calibration_matrix_columns_are_probability_distributions():
    noise_model = build_readout_noise_model(0.05)
    M = build_calibration_matrix(noise_model, num_qubits=4, shots=2048, seed_simulator=42)

    assert M.shape == (16, 16)
    np.testing.assert_allclose(M.sum(axis=0), np.ones(16), atol=1e-9)


def test_calibration_matrix_diagonal_dominant_for_moderate_noise():
    # At p=0.05 per-qubit bit-flip, correctly reading all 4 qubits
    # happens with probability (1-0.05)^4 ~= 0.815 -- the diagonal
    # should dominate each column.
    noise_model = build_readout_noise_model(0.05)
    M = build_calibration_matrix(noise_model, num_qubits=4, shots=4096, seed_simulator=42)

    for j in range(16):
        assert M[j, j] == M[:, j].max()


def test_apply_readout_mitigation_recovers_prepared_state():
    noise_model = build_readout_noise_model(0.10)
    M = build_calibration_matrix(noise_model, num_qubits=4, shots=4096, seed_simulator=42)

    # The noisy column for |0000> (index 0) is what was actually
    # measured when |0000> was prepared -- correcting it should
    # recover close to a delta function at index 0.
    noisy_probs = M[:, 0]
    corrected = apply_readout_mitigation(noisy_probs, M)

    assert corrected[0] > 0.99
    assert corrected.sum() == pytest.approx(1.0)
    assert np.all(corrected >= 0)


def test_fold_circuit_rejects_even_scale_factors():
    qc = QuantumCircuit(2)
    qc.h(0)

    with pytest.raises(ValueError):
        fold_circuit(qc, 2)


def test_fold_circuit_scale_1_is_unchanged():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    folded = fold_circuit(qc, 1)

    assert folded.depth() == qc.depth()


def test_fold_circuit_depth_scales_linearly():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    depths = [fold_circuit(qc, s).depth() for s in [1, 3, 5]]

    # Depth should roughly triple/quintuple relative to scale 1
    # (exact multiple depends on circuit structure, but must be
    # monotonically increasing and consistent with the (c-1)/2 folding
    # pairs formula).
    assert depths[0] < depths[1] < depths[2]
    assert depths[1] == pytest.approx(3 * depths[0], rel=0.2)


def test_fold_circuit_preserves_ideal_expectation_value():
    # The core correctness property ZNE relies on: folding must not
    # change the *noiseless* expectation value, only the amount of
    # noise accumulated under a real noise model.
    feature_map = create_feature_map(4, 2)
    ansatz = create_ansatz(4, 2)

    unitary = QuantumCircuit(4)
    unitary.compose(feature_map, inplace=True)
    unitary.compose(ansatz, inplace=True)

    rng = np.random.default_rng(0)
    bindings = dict(zip(feature_map.parameters, rng.uniform(-1, 1, size=len(feature_map.parameters))))
    bindings.update(dict(zip(ansatz.parameters, rng.uniform(-1, 1, size=len(ansatz.parameters)))))

    observable = SparsePauliOp.from_list([("ZIII", 1.0)])

    expectations = []
    for scale in [1, 3, 5]:
        folded = fold_circuit(unitary, scale)
        bound = folded.assign_parameters(bindings)
        state = Statevector(bound)
        expectations.append(state.expectation_value(observable).real)

    assert expectations[0] == pytest.approx(expectations[1], abs=1e-8)
    assert expectations[0] == pytest.approx(expectations[2], abs=1e-8)


def test_richardson_extrapolate_recovers_known_intercept():
    # y = 2 - 0.5*x -> intercept at x=0 is 2.0
    scale_factors = [1, 3, 5]
    expectations = [2 - 0.5 * s for s in scale_factors]

    result = richardson_extrapolate(scale_factors, expectations)

    assert result == pytest.approx(2.0, abs=1e-9)


def test_richardson_extrapolate_perfect_fit_zero_slope():
    scale_factors = [1, 3, 5]
    expectations = [0.4, 0.4, 0.4]

    result = richardson_extrapolate(scale_factors, expectations)

    assert result == pytest.approx(0.4, abs=1e-9)
