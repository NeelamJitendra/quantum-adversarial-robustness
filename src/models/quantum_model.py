"""
Quantum model definition for the Variational Quantum Classifier.

This module defines:
    1. The complete quantum circuit
    2. The observable(s) measured at the QNN output
    3. The EstimatorQNN
    4. The PyTorch-compatible quantum model

The implementation is designed for:
    Qiskit 2.5.1
    Qiskit Machine Learning 0.9.0
"""

from typing import List

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from src.circuits.feature_map import create_feature_map
from src.circuits.ansatz import create_ansatz


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SEED = 42

# "single_z" is the legacy observable used by the original 04A/04B
# notebooks (a single Z on the last qubit, e.g. "ZIII" for 4 qubits).
# Function-level defaults below preserve this behaviour so existing
# checkpoints/notebooks (e.g. 04C) keep working unchanged. New code
# (04D onward) should pass observable_mode="multi_z" explicitly -- see
# notebook/04D_phase0_diagnostics.ipynb for why.
DEFAULT_OBSERVABLE_MODE = "single_z"


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

def create_quantum_circuit(
    num_qubits=4,
    feature_reps=2,
    ansatz_reps=2,
):
    """
    Create the complete variational quantum circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    feature_reps : int
        Number of repetitions in the ZZFeatureMap.

    ansatz_reps : int
        Number of repetitions in the RealAmplitudes ansatz.

    Returns
    -------
    qc : QuantumCircuit
        Complete quantum circuit.

    feature_map : QuantumCircuit
        Data encoding circuit.

    ansatz : QuantumCircuit
        Trainable variational circuit.
    """

    feature_map = create_feature_map(
        num_qubits=num_qubits,
        reps=feature_reps,
    )

    ansatz = create_ansatz(
        num_qubits=num_qubits,
        reps=ansatz_reps,
    )

    qc = QuantumCircuit(num_qubits)

    qc.compose(
        feature_map,
        inplace=True,
    )

    qc.compose(
        ansatz,
        inplace=True,
    )

    return qc, feature_map, ansatz


# ============================================================
# OBSERVABLES
# ============================================================

def _pauli_label_for_qubit(qubit: int, num_qubits: int) -> str:
    """
    Build a single-qubit Z Pauli label in Qiskit's little-endian
    convention (leftmost character = highest qubit index), e.g.
    qubit=3, num_qubits=4 -> "ZIII"; qubit=0, num_qubits=4 -> "IIIZ".
    """

    position = num_qubits - 1 - qubit

    chars = ["I"] * num_qubits
    chars[position] = "Z"

    return "".join(chars)


def pauli_label_to_qubit(label: str) -> int:
    """
    Inverse of _pauli_label_for_qubit: given a single-qubit Pauli
    label (exactly one non-identity character), return the qubit
    index it measures.

    This is the single canonical place qubit indices are derived from
    an observable's Pauli string. Every other module that needs to
    know "which qubit does this observable measure" (noise-model
    inference, mitigation) should call get_measured_qubits() below
    instead of re-deriving bit-index conventions ad hoc -- notebook
    06A had two cells silently disagree on this (bitstring[-1] vs
    bitstring[0]) because the convention was re-derived in each place.
    """

    non_identity = [i for i, c in enumerate(label) if c != "I"]

    if len(non_identity) != 1:
        raise ValueError(
            f"Expected exactly one non-identity Pauli term, got {label!r}"
        )

    num_qubits = len(label)
    position = non_identity[0]

    return num_qubits - 1 - position


def create_observables(
    num_qubits: int = 4,
    observable_mode: str = DEFAULT_OBSERVABLE_MODE,
) -> List[SparsePauliOp]:
    """
    Build the observable(s) measured at the QNN output.

    Parameters
    ----------
    observable_mode : str
        "single_z": legacy single-qubit Z(q_{n-1}) readout used by the
            original 04A/04B notebooks (e.g. "ZIII" for 4 qubits).
            With only 2 reps of linear-entanglement RealAmplitudes,
            this forces all classification-relevant information to
            route through entanglement into a single qubit's
            expectation value -- suspected root cause of the ~58%
            accuracy found in 04A/04B (see 04D diagnostics).
        "multi_z": one Z observable per qubit. Gives the QNN a
            num_qubits-dimensional output instead of a single scalar,
            feeding a wider classical head. Default for the
            post-Phase-0-fix pipeline.

    Returns
    -------
    List[SparsePauliOp]
        One or num_qubits single-term observables.
    """

    if observable_mode == "single_z":
        label = _pauli_label_for_qubit(num_qubits - 1, num_qubits)
        return [SparsePauliOp.from_list([(label, 1.0)])]

    if observable_mode == "multi_z":
        return [
            SparsePauliOp.from_list(
                [(_pauli_label_for_qubit(qubit, num_qubits), 1.0)]
            )
            for qubit in range(num_qubits)
        ]

    raise ValueError(f"Unknown observable_mode: {observable_mode!r}")


def get_output_dim(
    num_qubits: int = 4,
    observable_mode: str = DEFAULT_OBSERVABLE_MODE,
) -> int:
    """Number of QNN outputs for a given observable_mode (1 for
    "single_z", num_qubits for "multi_z"). Use this to size the
    classical head, e.g.
    HybridClassifier(quantum_model, quantum_output_dim=get_output_dim(...)).
    """

    return len(
        create_observables(
            num_qubits=num_qubits,
            observable_mode=observable_mode,
        )
    )


def get_measured_qubits(qnn: EstimatorQNN) -> List[int]:
    """
    Canonical source of truth for "which qubit(s) does this QNN's
    observable measure", parsed directly from qnn.observables.
    """

    qubits = []

    for observable in qnn.observables:
        label = observable.paulis[0].to_label()
        qubits.append(pauli_label_to_qubit(label))

    return qubits


# ============================================================
# ESTIMATOR QNN
# ============================================================

def create_qnn(
    num_qubits=4,
    feature_reps=2,
    ansatz_reps=2,
    observable_mode=DEFAULT_OBSERVABLE_MODE,
    seed=DEFAULT_SEED,
):
    """
    Create the EstimatorQNN.

    Input gradients are explicitly enabled because
    FGSM requires gradients with respect to the input
    features.
    """

    qc, feature_map, ansatz = create_quantum_circuit(
        num_qubits=num_qubits,
        feature_reps=feature_reps,
        ansatz_reps=ansatz_reps,
    )

    observables = create_observables(
        num_qubits=num_qubits,
        observable_mode=observable_mode,
    )

    estimator = StatevectorEstimator(
        seed=seed
    )

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )

    return qnn


# ============================================================
# PYTORCH MODEL
# ============================================================

def create_model(
    num_qubits=4,
    feature_reps=2,
    ansatz_reps=2,
    observable_mode=DEFAULT_OBSERVABLE_MODE,
    seed=DEFAULT_SEED,
):
    """
    Create a PyTorch-compatible quantum model.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    observable_mode : str
        See create_observables(). Determines the QNN's output width
        (get_output_dim()) -- callers building a HybridClassifier on
        top of this model must size its classical head accordingly.

    seed : int
        Random seed for deterministic quantum evaluation AND for the
        ansatz's initial weights. TorchConnector draws its default
        initial weights from PyTorch's *global* RNG
        (`self._weights.data.uniform_(-1, 1)`), which our `seed` was
        previously never connected to -- two `create_model(seed=42)`
        calls produced genuinely different random initial weights,
        not just estimator sampling noise (confirmed via
        tests/test_reproducibility.py). We instead draw
        `initial_weights` explicitly from a local `np.random.Generator`
        seeded with `seed`, so model construction is reproducible
        independent of global RNG state / call order.

    Returns
    -------
    TorchConnector
        PyTorch-compatible quantum neural network.
    """

    qnn = create_qnn(
        num_qubits=num_qubits,
        feature_reps=feature_reps,
        ansatz_reps=ansatz_reps,
        observable_mode=observable_mode,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    initial_weights = rng.uniform(-1, 1, size=qnn.num_weights)

    model = TorchConnector(qnn, initial_weights=initial_weights)

    return model
