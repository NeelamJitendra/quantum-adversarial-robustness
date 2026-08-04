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
    ansatz_reps: int = 2
    # "single_z": legacy SparsePauliOp("ZIII") readout used by 04A/04B.
    # "multi_z": one Z observable per qubit, the Phase 0 fix default.
    observable_mode: str = "multi_z"


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.01
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
