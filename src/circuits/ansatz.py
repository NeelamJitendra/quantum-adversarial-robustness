"""
Variational ansatz definitions.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes


def create_ansatz(
    num_qubits: int = 4,
    reps: int = 2
) -> QuantumCircuit:
    """
    Uses the function-based real_amplitudes() rather than the
    RealAmplitudes class, which is deprecated as of Qiskit 2.1 and
    scheduled for removal in Qiskit 3.0.
    """
    return real_amplitudes(
        num_qubits=num_qubits,
        reps=reps,
        entanglement="linear"
    )