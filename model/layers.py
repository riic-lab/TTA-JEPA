import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = inputs.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return inputs / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        hidden = (int(8 / 3 * width) + 63) // 64 * 64
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
