"""
Aer NoiseModel construction for depolarizing, phase damping, and
readout noise (plus a combined depolarizing+readout condition).

Errors are attached to a fixed, explicit basis gate set
(rz/sx/x for single-qubit, cx for two-qubit) rather than the
un-transpiled circuit's native gates (p/ry/h/cx). The archived 06A
notebook applied noise to ["p", "ry", "h"] -- specific to that
circuit's exact pre-transpilation structure, which breaks silently if
the circuit changes (e.g. more ansatz reps decomposing differently).
Transpiling to a fixed basis first and applying noise there is both
more robust and closer to how noise is actually characterized on real
superconducting hardware (per-native-gate error rates on a fixed
basis).
"""

from typing import Optional

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, phase_damping_error

from src.noise.levels import NoiseLevel

BASIS_GATES = ["rz", "sx", "x", "cx"]
SINGLE_QUBIT_GATES = ["rz", "sx", "x"]
TWO_QUBIT_GATES = ["cx"]


def build_depolarizing_noise_model(p1: float, p2: float) -> NoiseModel:
    """1-qubit depolarizing error of probability p1 on rz/sx/x, and
    2-qubit depolarizing error of probability p2 on cx."""

    noise_model = NoiseModel(basis_gates=BASIS_GATES)

    if p1 > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p1, 1), SINGLE_QUBIT_GATES
        )

    if p2 > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p2, 2), TWO_QUBIT_GATES
        )

    return noise_model


def build_phase_damping_noise_model(gamma: float) -> NoiseModel:
    """Phase damping error of parameter gamma on every single-qubit
    gate (rz/sx/x)."""

    noise_model = NoiseModel(basis_gates=BASIS_GATES)

    if gamma > 0:
        noise_model.add_all_qubit_quantum_error(
            phase_damping_error(gamma), SINGLE_QUBIT_GATES
        )

    return noise_model


def build_readout_noise_model(p: float) -> NoiseModel:
    """Symmetric bit-flip readout error of probability p on every
    qubit's measurement."""

    noise_model = NoiseModel(basis_gates=BASIS_GATES)

    if p > 0:
        noise_model.add_all_qubit_readout_error(
            ReadoutError([[1 - p, p], [p, 1 - p]])
        )

    return noise_model


def build_combined_noise_model(p1: float, p2: float, readout_p: float) -> NoiseModel:
    """Depolarizing (1q + 2q) and readout error stacked in one
    NoiseModel."""

    noise_model = NoiseModel(basis_gates=BASIS_GATES)

    if p1 > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p1, 1), SINGLE_QUBIT_GATES
        )

    if p2 > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p2, 2), TWO_QUBIT_GATES
        )

    if readout_p > 0:
        noise_model.add_all_qubit_readout_error(
            ReadoutError([[1 - readout_p, readout_p], [readout_p, 1 - readout_p]])
        )

    return noise_model


def build_noise_model(noise_type: str, level: NoiseLevel) -> Optional[NoiseModel]:
    """
    Dispatch to the right builder given a noise_type
    ("none"|"depolarizing"|"phase_damping"|"readout"|"combined") and a
    level dict (see src/noise/levels.py:NOISE_LEVELS).

    Returns None for "none" -- pass directly as AerSimulator's
    noise_model argument (None means ideal simulation).
    """

    if noise_type == "none":
        return None

    if noise_type == "depolarizing":
        return build_depolarizing_noise_model(
            level["depolarizing_1q"], level["depolarizing_2q"]
        )

    if noise_type == "phase_damping":
        return build_phase_damping_noise_model(level["phase_damping"])

    if noise_type == "readout":
        return build_readout_noise_model(level["readout"])

    if noise_type == "combined":
        return build_combined_noise_model(
            level["depolarizing_1q"], level["depolarizing_2q"], level["readout"]
        )

    raise ValueError(f"Unknown noise_type: {noise_type!r}")
