"""
Quantum model definition for the Variational Quantum Classifier.

This module defines:
    1. The complete quantum circuit
    2. The EstimatorQNN
    3. The PyTorch-compatible quantum model

The implementation is designed for:
    Qiskit 2.5.1
    Qiskit Machine Learning 0.9.0
"""

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
# ESTIMATOR QNN
# ============================================================

def create_qnn(
    num_qubits=4,
    seed=DEFAULT_SEED,
):
    """
    Create the EstimatorQNN.

    Input gradients are explicitly enabled because
    FGSM requires gradients with respect to the input
    features.
    """

    qc, feature_map, ansatz = create_quantum_circuit(
        num_qubits=num_qubits
    )

    observable = SparsePauliOp.from_list(
        [
            ("ZIII", 1.0)
        ]
    )

    estimator = StatevectorEstimator(
        seed=seed
    )

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        observables=observable,
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
    seed=DEFAULT_SEED,
):
    """
    Create a PyTorch-compatible quantum model.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    seed : int
        Random seed for deterministic quantum evaluation.

    Returns
    -------
    TorchConnector
        PyTorch-compatible quantum neural network.
    """

    qnn = create_qnn(
        num_qubits=num_qubits,
        seed=seed,
    )

    model = TorchConnector(qnn)

    return model