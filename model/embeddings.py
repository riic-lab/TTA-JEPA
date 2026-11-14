import torch
from torch import nn
from torch.nn import functional as F


class TimeStepEncoder(nn.Module):
    def __init__(self, n_values: int, n_indicators: int, width: int):
        super().__init__()
        self.n_values = n_values
        self.n_indicators = n_indicators
        self.fc1 = nn.Linear(n_values + n_indicators, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, values: torch.Tensor, indicators: torch.Tensor) -> torch.Tensor:
        if values.shape[:-1] != indicators.shape[:-1]:
            raise ValueError("Values and indicators must share leading dimensions")
        if values.size(-1) != self.n_values or indicators.size(-1) != self.n_indicators:
            raise ValueError("Unexpected value or indicator width")
        features = torch.cat((values, indicators.to(values.dtype)), dim=-1)
        return F.silu(self.fc2(F.silu(self.fc1(features))))


class FourierPositionEncoding(nn.Module):
    def __init__(self, width: int, n_frequencies: int = 256,
                 hidden_dim: int = 32, gamma: float = 1.0,
                 output_std: float = 0.01):
        super().__init__()
        if n_frequencies < 1 or gamma <= 0 or output_std <= 0:
            raise ValueError("Fourier dimensions and initialization scales must be positive")
        self.n_frequencies = n_frequencies
        self.frequencies = nn.Parameter(torch.empty(n_frequencies))
        self.raw_layer_norm = nn.LayerNorm(2 * n_frequencies)
        self.projection1 = nn.Linear(2 * n_frequencies, hidden_dim)
        self.hidden_layer_norm = nn.LayerNorm(hidden_dim)
        self.projection2 = nn.Linear(hidden_dim, width)
        nn.init.normal_(self.frequencies, std=1.0 / gamma)
        nn.init.normal_(self.projection2.weight, std=output_std)
        nn.init.zeros_(self.projection2.bias)

    def forward(self, length: int, *, device, dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        angles = positions * self.frequencies.float()[None]
        raw = torch.cat((angles.cos(), angles.sin()), dim=-1) / self.n_frequencies**0.5
        hidden = F.gelu(self.projection1(self.raw_layer_norm(raw)))
        return self.projection2(self.hidden_layer_norm(hidden)).to(dtype)


class RotaryEmbedding(nn.Module):
    """Split-half RoPE convention; buffers are reconstructed, not checkpointed."""

    def __init__(self, head_dim: int, max_positions: int = 8192):
        super().__init__()
        if head_dim < 2 or head_dim % 2 or max_positions < 1:
            raise ValueError("RoPE requires a positive even head dimension and positive capacity")
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_positions).float(), theta)
        self.register_buffer("cosine", torch.cat((angles.cos(), angles.cos()), -1), persistent=False)
        self.register_buffer("sine", torch.cat((angles.sin(), angles.sin()), -1), persistent=False)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # inputs [B,heads,L,d]; positions [L] or [B,L].
        if positions.dtype != torch.long or positions.shape not in (
            inputs.shape[-2:-1], (inputs.size(0), inputs.size(-2))
        ):
            raise ValueError("Positions must be int64 [L] or [B,L]")
        if positions.numel() and (positions.min() < 0 or positions.max() >= len(self.cosine)):
            raise ValueError("Position exceeds RoPE capacity")
        cosine, sine = self.cosine[positions].to(inputs.dtype), self.sine[positions].to(inputs.dtype)
        if positions.ndim == 1:
            cosine, sine = cosine[None, None], sine[None, None]
        else:
            cosine, sine = cosine[:, None], sine[:, None]
        first, second = inputs.chunk(2, dim=-1)
        return inputs * cosine + torch.cat((-second, first), dim=-1) * sine
