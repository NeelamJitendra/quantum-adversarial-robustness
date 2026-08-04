"""
Quantum model definition for the Variational Quantum Classifier.

SPSA-gradient variant used for computational benchmarking.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import SPSAEstimatorGradient

from src.circuits.feature_map import create_feature_map
from src.circuits.ansatz import create_ansatz


def create_quantum_circuit(
    num_qubits: int = 4,
    feature_reps: int = 2,
    ansatz_reps: int = 2,
):
    """
    Create the complete quantum circuit.
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

    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    return qc, feature_map, ansatz


def create_qnn(
    num_qubits: int = 4,
):
    """
    Create an EstimatorQNN using SPSA gradients.
    """

    qc, feature_map, ansatz = create_quantum_circuit(
        num_qubits=num_qubits
    )

    observable = SparsePauliOp.from_list(
        [("ZIII", 1.0)]
    )

    estimator = StatevectorEstimator()

    # Explicit SPSA gradient
    gradient = SPSAEstimatorGradient(
        estimator=estimator
    )

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        gradient=gradient,
    )

    return qnn


def create_model(
    num_qubits: int = 4,
):
    """
    Create a PyTorch-compatible SPSA-gradient quantum model.
    """

    qnn = create_qnn(num_qubits)

    model = TorchConnector(qnn)

    return model