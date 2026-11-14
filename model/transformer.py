import math

import torch
from torch import nn
from torch.nn import functional as F

from .embeddings import RotaryEmbedding
from .layers import RMSNorm, SwiGLU


class ModernSelfAttention(nn.Module):
    def __init__(self, width: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        if min(width, n_heads, n_kv_heads) < 1 or width % n_heads or n_heads % n_kv_heads:
            raise ValueError("Width must divide into query heads, which must divide into KV groups")
        if (width // n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = width // n_heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def project(self, inputs, rotary, positions):
        batch, length, _ = inputs.shape

        def split(projection, heads):
            return projection.view(batch, length, heads, self.head_dim).transpose(1, 2)

        query = rotary(split(self.q_proj(inputs), self.n_heads), positions)
        key = rotary(split(self.k_proj(inputs), self.n_kv_heads), positions)
        value = split(self.v_proj(inputs), self.n_kv_heads)
        return query, key, value

    def forward(self, inputs, rotary, positions, allowed=None):
        query, key, value = self.project(inputs, rotary, positions)
        result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None if allowed is None else allowed[:, None],
            is_causal=allowed is None, dropout_p=0.0,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
        return self.o_proj(result.transpose(1, 2).contiguous().view_as(inputs))


class ModernTransformerBlock(nn.Module):
    def __init__(self, width: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.attn = ModernSelfAttention(width, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(width)
        self.mlp = SwiGLU(width)

    def forward(self, inputs, rotary, positions, allowed=None):
        inputs = inputs + self.attn(self.attn_norm(inputs), rotary, positions, allowed)
        return inputs + self.mlp(self.ffn_norm(inputs))


class ModernTransformer(nn.Module):
    """Right-padded [B,L,D] inputs. Custom True=allowed masks replace causality.

    Padded query outputs are not representations and must be excluded downstream.
    """

    def __init__(self, width: int, n_layers: int, n_heads: int,
                 n_kv_heads: int | None = None, max_positions: int = 8192):
        super().__init__()
        if n_layers < 1:
            raise ValueError("At least one layer is required")
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.width = width
        self.layers = nn.ModuleList([
            ModernTransformerBlock(width, n_heads, n_kv_heads) for _ in range(n_layers)
        ])
        self.rotary = RotaryEmbedding(width // n_heads, max_positions)
        self.final_norm = RMSNorm(width)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(layer.mlp.down_proj.weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, inputs, valid, allowed=None):
        if inputs.ndim != 3 or inputs.size(-1) != self.width:
            raise ValueError("Expected inputs [B,L,width]")
        if valid.dtype != torch.bool or valid.shape != inputs.shape[:2]:
            raise ValueError("Expected boolean validity [B,L]")
        if inputs.size(1) == 0 or (~valid[:, :-1] & valid[:, 1:]).any():
            raise ValueError("Inputs must be nonempty and right-padded")
        if allowed is not None:
            expected = (inputs.size(0), inputs.size(1), inputs.size(1))
            if allowed.dtype != torch.bool or allowed.shape != expected:
                raise ValueError("Expected boolean attention mask [B,L,L]")
            if (allowed & ~valid[:, None, :] & valid[:, :, None]).any():
                raise ValueError("Valid queries cannot attend padding keys")
        positions = torch.arange(inputs.size(1), device=inputs.device)
        for layer in self.layers:
            inputs = layer(inputs, self.rotary, positions, allowed)
        return self.final_norm(inputs)
