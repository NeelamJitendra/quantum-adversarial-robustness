"""
Zero-noise extrapolation (ZNE) via global unitary folding + linear
Richardson extrapolation (Temme, Bravyi, Gambetta 2017 -- matches
Mitiq's default LinearFactory approach). Hand-rolled rather than
adding mitiq as a new dependency this late in the timeline (same
reasoning as readout_calibration.py and the statistics module's
hand-rolled BH-FDR).

Applies to any noise type (unlike readout calibration, which targets
readout error specifically) -- folding amplifies whatever gate-level
noise is present in the noise model, so it's the natural mitigation
counterpart for depolarizing and phase damping conditions.
"""

from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.circuits.ansatz import create_ansatz
from src.circuits.feature_map import create_feature_map
from src.config import CircuitConfig
from src.models.quantum_model import create_qnn, get_measured_qubits
from src.noise.models import BASIS_GATES
from src.noise.observable_utils import counts_to_expectation


def fold_circuit(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """
    Global unitary folding: U -> U (U^dagger U)^((c-1)/2) for odd
    integer scale_factor c.

    The ideal (noiseless) output is unchanged, since U^dagger U is the
    identity -- but under a noise model, each additional gate
    application accumulates more physical noise. This is what lets ZNE
    probe "what would the output look like at c times the actual noise
    level" without altering the noise model itself.

    Parameters
    ----------
    circuit : QuantumCircuit
        The unitary part only (feature map + ansatz) -- must NOT
        already include measurement. Measurement is added by the
        caller after folding.
    """

    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(
            f"scale_factor must be a positive odd integer, got {scale_factor}"
        )

    folded = circuit.copy()

    if scale_factor == 1:
        return folded

    inverse = circuit.inverse()
    n_pairs = (scale_factor - 1) // 2

    for _ in range(n_pairs):
        folded.compose(inverse, inplace=True)
        folded.compose(circuit, inplace=True)

    return folded


def richardson_extrapolate(
    scale_factors: Sequence[float], expectations: Sequence[float]
) -> float:
    """
    Linear Richardson extrapolation to the zero-noise limit: fit a
    straight line to (scale_factor, expectation) pairs and return the
    fitted value at scale_factor=0 (the intercept).
    """

    scale_factors = np.asarray(scale_factors, dtype=float)
    expectations = np.asarray(expectations, dtype=float)

    slope, intercept = np.polyfit(scale_factors, expectations, deg=1)

    return float(intercept)


def zne_predict(
    quantum_weights: np.ndarray,
    classifier_weight: float,
    classifier_bias: float,
    X: np.ndarray,
    circuit_config: CircuitConfig,
    noise_model: Optional[NoiseModel],
    scale_factors: Sequence[int] = (1, 3, 5),
    shots: int = 1024,
    seed_simulator: int = 42,
) -> np.ndarray:
    """
    ZNE-corrected logits for a batch: fold the circuit at each scale
    factor, run all (sample, scale) circuits under noise_model in one
    batched Aer job, then Richardson-extrapolate each sample's
    per-scale Z-expectations to the zero-noise limit before applying
    the trained classical head.

    Only supports observable_mode="single_z", matching
    src.noise.aer_inference.evaluate_under_noise.
    """

    if circuit_config.observable_mode != "single_z":
        raise NotImplementedError(
            "zne_predict currently supports observable_mode='single_z' only"
        )

    feature_map = create_feature_map(
        num_qubits=circuit_config.num_qubits, reps=circuit_config.feature_reps
    )
    ansatz = create_ansatz(
        num_qubits=circuit_config.num_qubits, reps=circuit_config.ansatz_reps
    )

    unitary = QuantumCircuit(circuit_config.num_qubits)
    unitary.compose(feature_map, inplace=True)
    unitary.compose(ansatz, inplace=True)

    qnn = create_qnn(
        num_qubits=circuit_config.num_qubits,
        feature_reps=circuit_config.feature_reps,
        ansatz_reps=circuit_config.ansatz_reps,
        observable_mode=circuit_config.observable_mode,
    )
    qubit_index = get_measured_qubits(qnn)[0]

    simulator = AerSimulator(noise_model=noise_model)

    folded_transpiled = {}
    for scale in scale_factors:
        folded = fold_circuit(unitary, scale)
        folded.measure_all()
        folded_transpiled[scale] = transpile(folded, basis_gates=BASIS_GATES)

    all_circuits = []
    circuit_meta = []  # (sample_index, scale) per entry, same order as all_circuits

    for sample_index, row in enumerate(X):
        bindings = dict(zip(feature_map.parameters, row))
        bindings.update(dict(zip(ansatz.parameters, quantum_weights)))

        for scale in scale_factors:
            bound = folded_transpiled[scale].assign_parameters(bindings)
            all_circuits.append(bound)
            circuit_meta.append((sample_index, scale))

    result = simulator.run(
        all_circuits, shots=shots, seed_simulator=seed_simulator
    ).result()

    expectations_by_sample = {i: {} for i in range(len(X))}
    for job_index, (sample_index, scale) in enumerate(circuit_meta):
        counts = result.get_counts(job_index)
        z = counts_to_expectation(counts, qubit_index, circuit_config.num_qubits)
        expectations_by_sample[sample_index][scale] = z

    logits = np.empty(len(X))
    for sample_index in range(len(X)):
        zs = [expectations_by_sample[sample_index][s] for s in scale_factors]
        z_extrapolated = richardson_extrapolate(scale_factors, zs)
        logits[sample_index] = classifier_weight * z_extrapolated + classifier_bias

    return logits
