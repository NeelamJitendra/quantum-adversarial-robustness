"""
Readout error mitigation via a full 2**n x 2**n calibration matrix,
tractable to hand-roll at 4 qubits (16x16) -- rather than adding
qiskit-experiments as a new dependency this late in the timeline (same
reasoning as the noise/attacks/statistics modules' choices to hand-roll
rather than add dependencies).

Standard approach: prepare each computational basis state, measure it
under the noise model to learn P(measured | prepared), then invert
(pseudo-inverse) to correct an observed noisy distribution back toward
the ideal one.

Scope: this targets *readout* errors specifically (measurement
mis-assignment), not gate errors that occur during circuit execution.
Applied only to readout/combined noise conditions in the evaluation
sweep -- each mitigation technique should only be claimed to help the
noise type it targets, not depolarizing- or phase-damping-only
conditions.
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.circuits.ansatz import create_ansatz
from src.circuits.feature_map import create_feature_map
from src.config import CircuitConfig
from src.models.quantum_model import create_qnn, get_measured_qubits
from src.noise.models import BASIS_GATES
from src.noise.observable_utils import counts_to_probs, probs_to_expectation


def build_calibration_matrix(
    noise_model: Optional[NoiseModel],
    num_qubits: int = 4,
    shots: int = 4096,
    seed_simulator: int = 42,
) -> np.ndarray:
    """
    Build M where M[i, j] = P(measured bitstring index i | prepared
    basis state j), under noise_model, by preparing and measuring all
    2**num_qubits computational basis states.

    Uses the same integer bitstring-index convention as
    src.noise.observable_utils.counts_to_probs (bit position q from
    the LSB = qubit q).

    Returns
    -------
    np.ndarray, shape (2**num_qubits, 2**num_qubits)
    """

    dim = 2 ** num_qubits
    simulator = AerSimulator(noise_model=noise_model)

    circuits = []
    for j in range(dim):
        qc = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            if (j >> q) & 1:
                qc.x(q)
        qc.measure_all()
        circuits.append(qc)

    transpiled = transpile(circuits, simulator)
    result = simulator.run(
        transpiled, shots=shots, seed_simulator=seed_simulator
    ).result()

    M = np.zeros((dim, dim))

    for j in range(dim):
        counts = result.get_counts(j)
        for bitstring, count in counts.items():
            i = int(bitstring.replace(" ", ""), 2)
            M[i, j] = count / shots

    return M


def apply_readout_mitigation(
    probs: np.ndarray, calibration_matrix: np.ndarray
) -> np.ndarray:
    """
    Correct a noisy probability distribution via the pseudo-inverse of
    the calibration matrix.

    The pseudo-inverse solution isn't guaranteed to be a valid
    probability distribution (can have small negative entries or not
    sum to 1) -- clip negatives to 0 and renormalize, standard
    practice for matrix-inversion readout mitigation.
    """

    corrected = np.linalg.pinv(calibration_matrix) @ probs
    corrected = np.clip(corrected, 0, None)

    total = corrected.sum()
    if total > 0:
        corrected = corrected / total

    return corrected


def predict_with_calibration(
    quantum_weights: np.ndarray,
    classifier_weight: float,
    classifier_bias: float,
    X: np.ndarray,
    circuit_config: CircuitConfig,
    noise_model: Optional[NoiseModel],
    calibration_matrix: np.ndarray,
    shots: int = 1024,
    seed_simulator: int = 42,
) -> np.ndarray:
    """
    Logits for a batch, with readout-calibration correction applied to
    each sample's measured distribution before computing the
    Z-expectation. calibration_matrix should come from
    build_calibration_matrix() using the SAME noise_model (it only
    depends on the noise model, not the trained weights, so one
    calibration matrix can be reused across every seed's checkpoint
    for a given noise condition).
    """

    if circuit_config.observable_mode != "single_z":
        raise NotImplementedError(
            "predict_with_calibration currently supports observable_mode='single_z' only"
        )

    feature_map = create_feature_map(
        num_qubits=circuit_config.num_qubits, reps=circuit_config.feature_reps
    )
    ansatz = create_ansatz(
        num_qubits=circuit_config.num_qubits, reps=circuit_config.ansatz_reps
    )

    qc = QuantumCircuit(circuit_config.num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qc.measure_all()

    qnn = create_qnn(
        num_qubits=circuit_config.num_qubits,
        feature_reps=circuit_config.feature_reps,
        ansatz_reps=circuit_config.ansatz_reps,
        observable_mode=circuit_config.observable_mode,
    )
    qubit_index = get_measured_qubits(qnn)[0]

    simulator = AerSimulator(noise_model=noise_model)
    transpiled = transpile(qc, basis_gates=BASIS_GATES)

    bound_circuits = []
    for row in X:
        bindings = dict(zip(feature_map.parameters, row))
        bindings.update(dict(zip(ansatz.parameters, quantum_weights)))
        bound_circuits.append(transpiled.assign_parameters(bindings))

    result = simulator.run(
        bound_circuits, shots=shots, seed_simulator=seed_simulator
    ).result()

    logits = np.empty(len(X))
    for i in range(len(X)):
        counts = result.get_counts(i)
        raw_probs = counts_to_probs(counts, circuit_config.num_qubits)
        corrected_probs = apply_readout_mitigation(raw_probs, calibration_matrix)
        z = probs_to_expectation(corrected_probs, qubit_index, circuit_config.num_qubits)
        logits[i] = classifier_weight * z + classifier_bias

    return logits
