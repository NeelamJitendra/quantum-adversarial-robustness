"""
Canonical bit-parsing for converting Aer measurement counts into a
Z-expectation value for a specific qubit.

The archived 06A notebook had two cells silently disagree on which
qubit index the trained "ZIII" observable corresponds to
(bitstring[-1] vs bitstring[0]) because the convention was re-derived
ad hoc in each place. This is the one place that conversion happens
now; callers get the qubit index from
src.models.quantum_model.get_measured_qubits() (the canonical source
of truth for "which qubit does this observable measure") rather than
deriving it themselves.
"""

import numpy as np


def counts_to_probs(counts: dict, num_qubits: int) -> np.ndarray:
    """
    Convert an Aer counts dict into a length-2**num_qubits probability
    vector, indexed by the standard integer value of the bitstring
    (`int(bitstring, 2)`). This index convention exactly matches the
    little-endian qubit convention used everywhere else in this
    project: bit position q (from the LSB, `(index >> q) & 1`) equals
    qubit q's measurement outcome, since the bitstring's leftmost
    character (highest place value when read as an integer) is qubit
    num_qubits - 1, the same convention as pauli_label_to_qubit.
    """

    total = sum(counts.values())
    probs = np.zeros(2 ** num_qubits)

    if total == 0:
        return probs

    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        index = int(clean, 2)
        probs[index] = count / total

    return probs


def probs_to_expectation(probs: np.ndarray, qubit_index: int, num_qubits: int) -> float:
    """
    Compute <Z> for a specific qubit from a probability vector indexed
    the same way as counts_to_probs() -- used after readout-mitigation
    correction, where the corrected distribution is no longer backed
    by an integer counts dict.
    """

    expectation = 0.0

    for index, p in enumerate(probs):
        bit = (index >> qubit_index) & 1
        z_value = 1 if bit == 0 else -1
        expectation += z_value * p

    return float(expectation)


def counts_to_expectation(counts: dict, qubit_index: int, num_qubits: int) -> float:
    """
    Compute <Z> for a specific qubit from Aer measurement counts.

    Uses the same little-endian convention as
    src.models.quantum_model.pauli_label_to_qubit (rightmost bitstring
    character = qubit 0), which is Qiskit's standard convention for
    both Pauli operator strings and measurement bitstrings -- so a
    Pauli label position and a measurement bitstring position for the
    same qubit always agree.
    """

    total = sum(counts.values())

    if total == 0:
        return 0.0

    position = num_qubits - 1 - qubit_index
    expectation = 0.0

    for bitstring, count in counts.items():
        # Aer count keys can contain spaces when multiple classical
        # registers are present; measure_all() uses a single register,
        # but strip defensively.
        clean = bitstring.replace(" ", "")
        bit = clean[position]
        z_value = 1 if bit == "0" else -1
        expectation += z_value * count

    return expectation / total
