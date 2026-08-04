"""
Variational ansatz definitions.
"""

from qiskit.circuit.library import RealAmplitudes

def create_ansatz(
    num_qubits: int = 4,
    reps: int = 2
) -> RealAmplitudes:
    return RealAmplitudes(
        num_qubits=num_qubits,
        reps=reps,
        entanglement="linear"
    )