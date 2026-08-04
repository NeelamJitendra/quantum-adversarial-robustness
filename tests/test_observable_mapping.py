"""
Regression test for the qubit-indexing convention.

Notebook 06A had two cells silently disagree on which qubit index the
trained "ZIII" observable corresponds to (one read bitstring[-1] / q0,
another read bitstring[0] / q3). src/models/quantum_model.py's
get_measured_qubits() is the single canonical source of truth going
forward -- this test pins its behaviour and cross-checks it against
qiskit's own Statevector.expectation_value.
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src.models.quantum_model import (
    create_observables,
    create_qnn,
    get_measured_qubits,
    pauli_label_to_qubit,
)


def test_single_z_label_is_ZIII():
    observables = create_observables(num_qubits=4, observable_mode="single_z")
    labels = [obs.paulis[0].to_label() for obs in observables]
    assert labels == ["ZIII"]


def test_multi_z_labels_cover_all_qubits_low_to_high():
    observables = create_observables(num_qubits=4, observable_mode="multi_z")
    labels = [obs.paulis[0].to_label() for obs in observables]
    assert labels == ["IIIZ", "IIZI", "IZII", "ZIII"]


@pytest.mark.parametrize(
    "label,expected_qubit",
    [
        ("ZIII", 3),
        ("IZII", 2),
        ("IIZI", 1),
        ("IIIZ", 0),
    ],
)
def test_pauli_label_to_qubit(label, expected_qubit):
    assert pauli_label_to_qubit(label) == expected_qubit


def test_get_measured_qubits_single_z():
    qnn = create_qnn(observable_mode="single_z")
    assert get_measured_qubits(qnn) == [3]


def test_get_measured_qubits_multi_z():
    qnn = create_qnn(observable_mode="multi_z")
    assert get_measured_qubits(qnn) == [0, 1, 2, 3]


def test_ZIII_expectation_matches_statevector_on_qubit_3():
    """
    Cross-validate the "ZIII" = Z-on-qubit-3 convention against an
    independent, ground-truth calculation via Statevector, on a
    circuit with no special symmetry (a fixed rotation on every
    qubit) so the check isn't vacuous.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.rx(0.3, 0)
    qc.rx(0.7, 1)
    qc.rx(1.1, 2)
    qc.rx(1.9, 3)

    state = Statevector(qc)

    observables = create_observables(num_qubits=4, observable_mode="single_z")
    expected_from_op = state.expectation_value(observables[0]).real

    # Independent ground truth: Z-expectation on qubit 3 alone is
    # cos(theta) for a bare Rx(theta) rotation with no entanglement.
    expected_from_physics = np.cos(1.9)

    assert np.isclose(expected_from_op, expected_from_physics, atol=1e-8)
