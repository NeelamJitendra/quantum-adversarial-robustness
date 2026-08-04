"""
Feature map definitions for the Variational Quantum Classifier.
"""

from qiskit.circuit.library import ZZFeatureMap


def create_feature_map(
    num_qubits: int = 4,
    reps: int = 2
) -> ZZFeatureMap:
    """
    Create the quantum feature map.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    reps : int
        Number of repetitions.

    Returns
    -------
    ZZFeatureMap
    """

    return ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=reps
    )