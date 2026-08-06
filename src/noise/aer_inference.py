"""
Shot-based noisy inference for a trained VQC checkpoint.

Binds a checkpoint's ansatz weights and a batch of input features into
the measurement circuit, runs it under a given Aer noise model
(None = ideal/noiseless), and reconstructs logits through the same
trained classical head (single Linear layer) the model was trained
with. This is a cleaned-up, batched port of the archived 06A
notebook's working-but-messy evaluation loop.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.circuits.ansatz import create_ansatz
from src.circuits.feature_map import create_feature_map
from src.config import CircuitConfig
from src.models.quantum_model import create_qnn, get_measured_qubits
from src.noise.models import BASIS_GATES
from src.noise.observable_utils import counts_to_expectation


def build_measurement_circuit(
    num_qubits: int, feature_reps: int, ansatz_reps: int
) -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """Feature map + ansatz + measure_all(), matching the circuit the
    VQC was trained on but with an explicit measurement for shot-based
    (Aer) inference instead of exact (StatevectorEstimator) evaluation."""

    feature_map = create_feature_map(num_qubits=num_qubits, reps=feature_reps)
    ansatz = create_ansatz(num_qubits=num_qubits, reps=ansatz_reps)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qc.measure_all()

    return qc, feature_map, ansatz


def load_checkpoint_params(checkpoint_path, quantum_output_dim: int = 1):
    """
    Extract (quantum_weights, classifier_weight, classifier_bias) from
    a saved HybridClassifier state_dict.

    Only supports observable_mode="single_z" (quantum_output_dim=1) --
    aer_inference reconstructs the scalar Z-expectation directly from
    counts; a multi_z checkpoint would need one counts_to_expectation
    call per output qubit, not currently implemented.
    """

    if quantum_output_dim != 1:
        raise NotImplementedError(
            "aer_inference currently supports single_z (quantum_output_dim=1) only"
        )

    state_dict = torch.load(checkpoint_path, weights_only=True)

    quantum_weights = state_dict["quantum.weight"].numpy()
    classifier_weight = state_dict["classifier.weight"].item()
    classifier_bias = state_dict["classifier.bias"].item()

    return quantum_weights, classifier_weight, classifier_bias


def evaluate_under_noise(
    quantum_weights: np.ndarray,
    classifier_weight: float,
    classifier_bias: float,
    X: np.ndarray,
    circuit_config: CircuitConfig,
    noise_model: Optional[NoiseModel],
    shots: int = 1024,
    seed_simulator: int = 42,
) -> np.ndarray:
    """
    Run X through the trained circuit under noise_model (None = ideal)
    and return an array of logits (classifier_weight * <Z> + bias),
    one per row of X.

    All rows are submitted to Aer as a single batched job (one
    simulator.run() call over a list of bound circuits) rather than
    one call per sample -- Aer's per-call dispatch overhead dominates
    wall-clock time at this qubit count, so batching matters.
    """

    if circuit_config.observable_mode != "single_z":
        raise NotImplementedError(
            "aer_inference currently supports observable_mode='single_z' only"
        )

    qc, feature_map, ansatz = build_measurement_circuit(
        circuit_config.num_qubits, circuit_config.feature_reps, circuit_config.ansatz_reps
    )

    # Canonical qubit index for this observable -- derived once from
    # the same construction used at training time, not re-derived here.
    qnn = create_qnn(
        num_qubits=circuit_config.num_qubits,
        feature_reps=circuit_config.feature_reps,
        ansatz_reps=circuit_config.ansatz_reps,
        observable_mode=circuit_config.observable_mode,
    )
    measured_qubits = get_measured_qubits(qnn)
    qubit_index = measured_qubits[0]

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
        z = counts_to_expectation(counts, qubit_index, circuit_config.num_qubits)
        logits[i] = classifier_weight * z + classifier_bias

    return logits
