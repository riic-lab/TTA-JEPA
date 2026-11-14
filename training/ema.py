from copy import deepcopy

import torch
from torch import nn


class EMATarget(nn.Module):
    """Copy an explicitly selected source, not necessarily the student.

    Update after the source optimizer step. Buffers are copied, not averaged.
    """

    def __init__(self, source: nn.Module, momentum: float = 0.996):
        super().__init__()
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must lie in [0,1]")
        self.momentum = momentum
        self.target = deepcopy(source).requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    @torch.no_grad()
    def update(self, source: nn.Module):
        target_params = dict(self.target.named_parameters())
        source_params = dict(source.named_parameters())
        target_buffers = dict(self.target.named_buffers())
        source_buffers = dict(source.named_buffers())
        # Validate the complete pairing before mutating any target parameter.
        for target, online in ((target_params, source_params), (target_buffers, source_buffers)):
            if target.keys() != online.keys():
                raise ValueError("EMA source and target structures differ")
            for name, value in target.items():
                other = online[name]
                if (value.shape, value.dtype, value.device) != (other.shape, other.dtype, other.device):
                    raise ValueError(f"EMA tensor mismatch: {name}")
        for name, value in target_params.items():
            value.lerp_(source_params[name].detach(), 1.0 - self.momentum)
        for name, value in target_buffers.items():
            value.copy_(source_buffers[name])
