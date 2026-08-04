"""
Feature map definitions for the Variational Quantum Classifier.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map


def create_feature_map(
    num_qubits: int = 4,
    reps: int = 2
) -> QuantumCircuit:
    """
    Create the quantum feature map.

    Uses the function-based zz_feature_map() rather than the
    ZZFeatureMap class, which is deprecated as of Qiskit 2.1 and
    scheduled for removal in Qiskit 3.0.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    reps : int
        Number of repetitions.

    Returns
    -------
    QuantumCircuit
    """

    return zz_feature_map(
        feature_dimension=num_qubits,
        reps=reps
    )