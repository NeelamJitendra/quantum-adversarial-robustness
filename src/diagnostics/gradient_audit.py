"""
Phase 0 diagnostic steps 1-2: confirm gradients actually flow to
every trainable parameter, and that the optimizer is tracking all of
them, before assuming a training-dynamics or expressivity problem.

See notebook/04D_phase0_diagnostics.ipynb.
"""

from typing import List

import torch
import torch.nn as nn


def audit_gradients(
    model: nn.Module,
    criterion: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> List[dict]:
    """
    Run one real forward/backward pass and report per-parameter
    gradient health.

    Returns
    -------
    list[dict]
        One entry per named parameter: name, shape, grad_is_none,
        grad_norm.
    """

    model.zero_grad()

    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()

    report = []

    for name, param in model.named_parameters():
        grad_is_none = param.grad is None
        grad_norm = None if grad_is_none else param.grad.norm().item()

        report.append({
            "name": name,
            "shape": tuple(param.shape),
            "grad_is_none": grad_is_none,
            "grad_norm": grad_norm,
        })

    return report


def check_optimizer_registration(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """
    Confirm the optimizer is tracking exactly the parameters
    model.parameters() reports (rules out a silently frozen or
    unregistered parameter).
    """

    model_param_count = sum(p.numel() for p in model.parameters())

    optimizer_param_count = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
    )

    return {
        "model_param_count": model_param_count,
        "optimizer_param_count": optimizer_param_count,
        "match": model_param_count == optimizer_param_count,
    }
