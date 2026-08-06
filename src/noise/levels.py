"""
Named noise-level sweep table -- the single source of truth for noise
intensity, replacing scattered constants like the archived 06A
notebook's module-level P_1Q = 0.001 / P_2Q = 0.01 / P_READOUT = 0.02.

Levels are kept comparable in order-of-magnitude across noise types
(L2 matches 06A's original defaults) so a comparison across noise
*types* at the same level isn't confounded by mismatched severity.
Numbers are illustrative/order-of-magnitude, matching typical
superconducting-qubit gate/readout error rates -- cite a specific
backend or paper for the exact digits used in the thesis text rather
than presenting these as measured.

Core experiment: L0-L3. L4 is a stress-test, explicitly droppable if
time runs short (see the thesis architecture plan).
"""

from typing import TypedDict


class NoiseLevel(TypedDict):
    depolarizing_1q: float
    depolarizing_2q: float
    phase_damping: float
    readout: float


NOISE_LEVELS: dict = {
    "L0": {"depolarizing_1q": 0.0,    "depolarizing_2q": 0.0,   "phase_damping": 0.0,   "readout": 0.0},
    "L1": {"depolarizing_1q": 0.0005, "depolarizing_2q": 0.005, "phase_damping": 0.005, "readout": 0.01},
    "L2": {"depolarizing_1q": 0.001,  "depolarizing_2q": 0.01,  "phase_damping": 0.01,  "readout": 0.02},
    "L3": {"depolarizing_1q": 0.005,  "depolarizing_2q": 0.03,  "phase_damping": 0.03,  "readout": 0.05},
    "L4": {"depolarizing_1q": 0.01,   "depolarizing_2q": 0.05,  "phase_damping": 0.05,  "readout": 0.10},
}

CORE_LEVELS = ["L0", "L1", "L2", "L3"]
STRETCH_LEVELS = ["L4"]

NOISE_TYPES = ["none", "depolarizing", "phase_damping", "readout", "combined"]
