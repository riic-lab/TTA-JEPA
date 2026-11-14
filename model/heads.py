from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class PrefixTaskHeads(nn.Module):
    """One independent head per ordered prefix; returns [..., number_of_tasks]."""

    def __init__(self, width: int, horizons: Mapping[str, int],
                 hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons.values()):
            raise ValueError("Task horizons must be positive integers")
        self.width = width
        self.task_names = tuple(horizons)
        self.horizons = tuple(horizons.values())
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(h * width), nn.Linear(h * width, hidden_dim),
                          nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            for h in self.horizons
        ])
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, tokens):
        if tokens.ndim < 3 or tokens.size(-1) != self.width or tokens.size(-2) < max(self.horizons):
            raise ValueError("Expected [...,H,width] covering every task horizon")
        return torch.cat([
            head(tokens[..., :h, :].flatten(-2))
            for head, h in zip(self.heads, self.horizons, strict=True)
        ], dim=-1)


class CLSTaskHeads(nn.Module):
    """Shared input LayerNorm with independent task MLP parameters."""

    def __init__(self, width: int, tasks: Sequence[str],
                 hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError("Task names must be nonempty and unique")
        self.width, self.task_names = width, tuple(tasks)
        self.norm = nn.LayerNorm(width)
        self.weight1 = nn.Parameter(torch.empty(len(tasks), width, hidden_dim))
        self.bias1 = nn.Parameter(torch.zeros(len(tasks), hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(len(tasks), hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(len(tasks)))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight1, std=0.02)
        nn.init.normal_(self.weight2, std=0.02)

    def forward(self, embedding):
        if embedding.ndim < 2 or embedding.size(-1) != self.width:
            raise ValueError("Expected [...,width] embeddings")
        hidden = F.silu(torch.einsum("...d,kdh->...kh", self.norm(embedding), self.weight1) + self.bias1)
        return torch.einsum("...kh,kh->...k", self.dropout(hidden), self.weight2) + self.bias2
