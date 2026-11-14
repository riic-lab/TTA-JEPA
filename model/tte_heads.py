import torch
from torch import nn
from torch.nn import functional as F


class TTEHeads(nn.Module):
    """Variable-specific weights shared across horizons; output [B,A,K,H]."""

    def __init__(self, n_targets, width=256, hidden_dim=512,
                 n_frequencies=8, identity_dim=16):
        super().__init__()
        self.n_targets = n_targets
        self.frequencies = nn.Parameter(1.5 ** torch.arange(n_frequencies).float())
        self.target_embedding = nn.Embedding(n_targets, identity_dim)
        conditioning_dim = 2 * n_frequencies + identity_dim
        self.state_weight = nn.Parameter(torch.empty(n_targets, width, hidden_dim))
        self.condition_weight = nn.Parameter(torch.empty(n_targets, conditioning_dim, hidden_dim))
        self.bias1 = nn.Parameter(torch.zeros(n_targets, hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(n_targets, hidden_dim))
        self.bias2 = nn.Parameter(torch.full((n_targets,), -5.0))
        for p in (self.target_embedding.weight, self.state_weight,
                  self.condition_weight, self.weight2):
            nn.init.normal_(p, std=0.02)

    def conditioning(self, threshold):
        if threshold.ndim != 3 or threshold.size(-1) != self.n_targets:
            raise ValueError("Threshold must have shape [B,A,K]")
        # Thresholds use the original [-4,4] reference interval; do not clamp.
        angles = (2 * (threshold + 4) / 8 - 1)[..., None] * self.frequencies
        identity = self.target_embedding.weight.expand(*threshold.shape[:-1], -1, -1)
        return torch.cat((angles.sin(), angles.cos(), identity), -1)

    def forward(self, states, threshold):
        condition = self.conditioning(threshold)
        if states.ndim != 4 or states.shape[:2] != threshold.shape[:2]:
            raise ValueError("States must have shape [B,A,H,D]")
        hidden = F.silu(torch.einsum("bahd,kdf->bahkf", states, self.state_weight)
                        + torch.einsum("bakc,kcf->bakf", condition, self.condition_weight)[:, :, None]
                        + self.bias1)
        return (torch.einsum("bahkf,kf->bahk", hidden, self.weight2) + self.bias2).permute(0, 1, 3, 2)
