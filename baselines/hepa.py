import torch
from torch import nn
from model.embeddings import TimeStepEncoder
from model.transformer import ModernTransformer


class HEPA(nn.Module):
    """Shared trunk: causal history, bidirectional future, no EMA or stop-gradient."""

    def __init__(self, dimensions, width, n_layers, n_heads, n_kv_heads, max_positions, horizon):
        super().__init__()
        self.width, self.n_tokens = width, 1
        self.timestep = TimeStepEncoder(dimensions["n_values"], dimensions["n_indicators"], width)
        self.static = nn.Linear(dimensions["n_static"], width)
        self.trunk = ModernTransformer(width, n_layers, n_heads, n_kv_heads, max_positions)
        self.fusion = nn.LayerNorm(width)
        self.pool_query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, n_heads, batch_first=True, dropout=0.0)
        self.predictor = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, width),
                                       nn.GELU(), nn.Linear(width, width))
        self.event_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, static, values, indicators, valid, anchors, anchor_valid):
        sequence = self.trunk(self.timestep(values, indicators), valid)
        rows = torch.arange(len(values), device=values.device)[:, None]
        states = self.fusion(sequence[rows, anchors] + self.static(static)[:, None])
        return states[:, :, None] * anchor_valid[..., None, None]

    def pretrain(self, batch):
        anchors = batch["history_valid"].sum(-1, keepdim=True) - 1
        history = self(batch["static"], batch["history"], batch["history_indicators"],
                       batch["history_valid"], anchors, torch.ones_like(anchors, dtype=torch.bool))[:, 0, 0]
        prediction = self.predictor(torch.cat((history, batch["delta"][:, None]), -1))
        valid = batch["future_valid"]
        tokens = self.timestep(batch["future"], batch["future_indicators"])
        sequence = self.trunk(tokens, valid, valid[:, None].expand(-1, valid.size(1), -1))
        target, _ = self.pool(self.pool_query.expand(len(tokens), -1, -1), sequence, sequence,
                              key_padding_mask=~valid, need_weights=False)
        return prediction, target[:, 0]
