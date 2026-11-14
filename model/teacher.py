import torch
from torch import nn

from .embeddings import TimeStepEncoder
from .transformer import ModernTransformer


class FutureTeacher(nn.Module):
    """One causal block over hourly observations; no static or query inputs."""

    def __init__(self, n_values, n_indicators, width=256, n_heads=4, horizon=48):
        super().__init__()
        self.horizon = horizon
        self.time_step_encoder = TimeStepEncoder(n_values, n_indicators, width)
        self.transformer = ModernTransformer(width, 1, n_heads, n_heads, max_positions=horizon)

    def forward(self, future, indicators, valid):
        if future.ndim != 4 or future.shape[:-1] != valid.shape or future.size(2) != self.horizon:
            raise ValueError("Expected future [B,A,H,C] and validity [B,A,H]")
        tokens = self.time_step_encoder(future, indicators)
        flat_valid = valid.flatten(0, 1)
        tokens = tokens.flatten(0, 1) * flat_valid[..., None]
        allowed = torch.ones(self.horizon, self.horizon, dtype=torch.bool,
                             device=tokens.device).tril()[None] & flat_valid[:, None]
        states = self.transformer(tokens, flat_valid, allowed).unflatten(0, future.shape[:2])
        # Preserve all horizon readouts for nonempty windows: alignment covers
        # all 48 horizons, while the TTE loss has its own censoring mask.
        return states * valid.any(-1)[..., None, None]
