import torch
from torch import nn
from torch.nn import functional as F

from .embeddings import TimeStepEncoder
from .layers import ProjectionMLP
from .transformer import ModernTransformer


class HistoryQueryEncoder(nn.Module):
    """Shared causal history stream; each anchor has a separate query stream.

    Queries have identity rotary positions and unrestricted within-anchor
    query attention. History never attends queries or future observations.
    """

    def __init__(self, n_static, n_values, n_indicators, width=512, n_layers=3,
                 n_heads=8, n_kv_heads=8, horizon=48, max_positions=8192):
        super().__init__()
        self.width, self.horizon = width, horizon
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.time_step_encoder = TimeStepEncoder(n_values, n_indicators, width)
        self.query_embedding = nn.Embedding(horizon, width)
        self.sequence_encoder = ModernTransformer(width, n_layers, n_heads, n_kv_heads, max_positions)
        nn.init.normal_(self.query_embedding.weight, std=0.02)

    def forward(self, static, history, indicators, valid, anchors, anchor_valid):
        if history.ndim != 3 or history.size(1) == 0:
            raise ValueError("History must have nonempty shape [B,T,C]")
        if valid.dtype != torch.bool or valid.shape != history.shape[:2]:
            raise ValueError("History validity must be boolean [B,T]")
        if (~valid[:, :-1] & valid[:, 1:]).any():
            raise ValueError("History must be right-padded")
        if anchors.ndim != 2 or anchors.dtype != torch.long or anchors.size(0) != history.size(0):
            raise ValueError("Anchors must be int64 [B,A], zero-based history offsets")
        if anchor_valid.dtype != torch.bool or anchor_valid.shape != anchors.shape:
            raise ValueError("Anchor validity must be boolean [B,A]")
        if (anchor_valid & ((anchors < 0) | (anchors >= valid.sum(-1)[:, None]))).any():
            raise ValueError("A valid anchor must refer to observed history")
        patient = torch.cat((self.static_encoder(static)[:, None],
                             self.time_step_encoder(history, indicators)), dim=1)
        patient_valid = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)
        cutoff = torch.where(anchor_valid, anchors + 1, 0)
        result = self._encode(patient, patient_valid, cutoff)
        return result * anchor_valid[..., None, None]

    def _encode(self, patient, valid, cutoff):
        batch, anchors = cutoff.shape
        queries = self.query_embedding.weight[None, None].expand(batch, anchors, -1, -1).clone()
        encoder = self.sequence_encoder
        positions = torch.arange(patient.size(1), device=patient.device)
        query_positions = torch.zeros(self.horizon, device=patient.device, dtype=torch.long)
        patient_allowed = torch.ones(patient.size(1), patient.size(1),
                                     dtype=torch.bool, device=patient.device).tril()[None] & valid[:, None]
        patient_keys = (positions[None, None, None] <= cutoff[..., None, None]) & valid[:, None, None]
        patient_keys = patient_keys.expand(-1, -1, self.horizon, -1)
        query_keys = torch.ones(batch, anchors, self.horizon, self.horizon,
                               dtype=torch.bool, device=patient.device)
        allowed = torch.cat((query_keys, patient_keys), -1).reshape(batch * anchors, 1, self.horizon, -1)
        for block in encoder.layers:
            patient_q, patient_k, patient_v = block.attn.project(block.attn_norm(patient), encoder.rotary, positions)
            flat = queries.reshape(batch * anchors, self.horizon, self.width)
            query_q, query_k, query_v = block.attn.project(block.attn_norm(flat), encoder.rotary, query_positions)
            # Fan out only K/V; history computation is shared across anchors.
            repeated_k = patient_k[:, None].expand(-1, anchors, -1, -1, -1).flatten(0, 1)
            repeated_v = patient_v[:, None].expand(-1, anchors, -1, -1, -1).flatten(0, 1)
            gqa = block.attn.n_heads != block.attn.n_kv_heads
            patient_update = F.scaled_dot_product_attention(
                patient_q, patient_k, patient_v, attn_mask=patient_allowed[:, None], enable_gqa=gqa)
            query_update = F.scaled_dot_product_attention(
                query_q, torch.cat((query_k, repeated_k), -2), torch.cat((query_v, repeated_v), -2),
                attn_mask=allowed, enable_gqa=gqa)
            patient = patient + block.attn.o_proj(patient_update.transpose(1, 2).contiguous().view_as(patient))
            flat = flat + block.attn.o_proj(query_update.transpose(1, 2).contiguous().view_as(flat))
            patient = patient + block.mlp(block.ffn_norm(patient))
            queries = (flat + block.mlp(block.ffn_norm(flat))).view(batch, anchors, self.horizon, self.width)
        index = torch.arange(batch, device=patient.device)[:, None]
        return encoder.final_norm(queries + patient[index, cutoff][:, :, None])


class HistoryRepresentation(nn.Module):
    def __init__(self, encoder, output_dim=256):
        super().__init__()
        self.encoder = encoder
        self.adapter = ProjectionMLP(encoder.width, encoder.width, encoder.width)
        self.output = nn.Linear(encoder.width, output_dim)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def project(self, queries):
        return self.output(self.adapter(queries))

    def forward(self, batch):
        return self.project(self.encoder(
            batch["static"], batch["history"], batch["history_indicators"],
            batch["history_valid"], batch["anchors"], batch["anchor_valid"]))
