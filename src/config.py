"""
Experiment configuration.

Centralizes hyperparameters that were previously scattered as
module-level constants across notebooks (e.g. 06A's
P_1Q = 0.001 / P_2Q = 0.01 / P_READOUT = 0.02) into JSON-serializable
dataclasses, so every seed/circuit/noise/attack/mitigation combination
run in the experiment grid has a single, inspectable, reproducible
record on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class CircuitConfig:
    num_qubits: int = 4
    feature_reps: int = 2
    # Phase 0 diagnostics (notebook/04_phase0_diagnostics.ipynb) found
    # accuracy plateaus regardless of learning rate, batching, or
    # observable width (single_z vs multi_z performed the same), but
    # improves with ansatz depth: reps=2 plateaus at ~58%, reps=4 at
    # ~63-64%. Accepted as the working default rather than pursuing a
    # full architecture redesign (data re-uploading) -- see 04D for
    # the full investigation and the tradeoff decision.
    ansatz_reps: int = 4
    # "single_z" (legacy SparsePauliOp("ZIII")) performs the same as
    # "multi_z" (one Z per qubit) -- observable width was ruled out as
    # the bottleneck in 04D Step 6d, so the simpler single_z is default.
    observable_mode: str = "single_z"


@dataclass
class TrainConfig:
    # 20 epochs matches the reference convergence run (04D Step 6f):
    # validation accuracy plateaus by epoch ~4 and oscillates without
    # further gains through epoch 20, so this is enough budget to reach
    # the observed ceiling without wasting compute past it.
    epochs: int = 20
    batch_size: int = 32
    # 0.01 (the original 04A/04B value) was confirmed too low to
    # escape the loss plateau within any practical epoch budget (04D
    # Step 6a); 0.05 is the validated working value.
    lr: float = 0.05
    optimizer: str = "adam"


@dataclass
class NoiseConfig:
    # "none" | "depolarizing" | "phase_damping" | "readout" | "combined"
    noise_type: str = "none"
    # Key into src/noise/levels.py NOISE_LEVELS, e.g. "L0".."L4".
    level: str = "L0"
    shots: int = 1024


@dataclass
class AttackConfig:
    method: str = "fgsm"   # "fgsm" | "pgd"
    epsilon: float = 0.0
    norm: str = "inf"      # matches ART's `norm` argument convention


@dataclass
class MitigationConfig:
    method: Optional[str] = None  # None | "readout_calibration" | "zne"


@dataclass
class ExperimentConfig:
    seed: int
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path: Union[str, Path]) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text())
        return ExperimentConfig(
            seed=data["seed"],
            circuit=CircuitConfig(**data.get("circuit", {})),
            train=TrainConfig(**data.get("train", {})),
            noise=NoiseConfig(**data.get("noise", {})),
            attack=AttackConfig(**data.get("attack", {})),
            mitigation=MitigationConfig(**data.get("mitigation", {})),
        )
