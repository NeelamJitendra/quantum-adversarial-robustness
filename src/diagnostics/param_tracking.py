"""
Phase 0 diagnostic step 3: track how far each parameter moves from
its initialization over training, to distinguish "the quantum branch
isn't learning" from "only the classical head is learning" (both can
produce a loss plateau near ln(2), but only one implicates the
quantum circuit itself).

See notebook/04D_phase0_diagnostics.ipynb.
"""

from typing import Dict

import torch.nn as nn


class ParameterMovementTracker:
    """Snapshots a model's parameters at construction time and reports
    L2 distance moved from that snapshot on demand."""

    def __init__(self, model: nn.Module):
        self._initial = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    def movement(self, model: nn.Module) -> Dict[str, float]:
        """L2 distance of each named parameter from its value at
        tracker construction time."""

        return {
            name: (param.detach() - self._initial[name]).norm().item()
            for name, param in model.named_parameters()
        }
