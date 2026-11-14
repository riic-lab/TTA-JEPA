import torch
from torch import nn
from torch.nn import functional as F


class CompactTTEHeads(nn.Module):
    """Independent target heads emitting all horizons from one anchor state."""

    def __init__(self, n_targets, width, hidden_dim, horizon):
        super().__init__()
        self.frequencies = nn.Parameter(1.5 ** torch.arange(8, dtype=torch.float32))
        self.identity = nn.Parameter(torch.empty(n_targets, 16))
        self.state_weight = nn.Parameter(torch.empty(n_targets, width, hidden_dim))
        self.condition_weight = nn.Parameter(torch.empty(n_targets, 32, hidden_dim))
        self.bias1 = nn.Parameter(torch.zeros(n_targets, hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(n_targets, hidden_dim, horizon))
        self.bias2 = nn.Parameter(torch.full((n_targets, horizon), -5.0))
        for p in (self.identity, self.state_weight, self.condition_weight, self.weight2):
            nn.init.normal_(p, std=0.02)

    def forward(self, states, threshold):
        angle = threshold[..., None] / 4 * self.frequencies
        identity = self.identity.expand(*threshold.shape[:-1], -1, -1)
        condition = torch.cat((angle.sin(), angle.cos(), identity), -1)
        hidden = F.silu(torch.einsum("...d,kdh->...kh", states, self.state_weight)
                        + torch.einsum("...kc,kch->...kh", condition, self.condition_weight) + self.bias1)
        return torch.einsum("...kh,khr->...kr", hidden, self.weight2) + self.bias2
