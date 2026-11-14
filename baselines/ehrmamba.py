import math
import torch
from torch import nn
from model.embeddings import TimeStepEncoder
from model.layers import RMSNorm
from model.compact_tte import CompactTTEHeads


class MambaBlock(nn.Module):
    def __init__(self, width, depth, index):
        super().__init__()
        from mamba_ssm import Mamba3
        self.norm = RMSNorm(width)
        self.mixer = Mamba3(d_model=width, d_state=64, expand=2, headdim=64,
                            chunk_size=64, dropout=0.0, is_mimo=False, layer_idx=index, n_layer=depth)
        with torch.no_grad():
            self.mixer.out_proj.weight.div_(math.sqrt(depth))

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class EHRMamba(nn.Module):
    def __init__(self, dimensions, width, n_layers, horizon, tte_hidden):
        super().__init__()
        self.width, self.n_tokens = width, 1
        self.static = nn.Sequential(nn.Linear(dimensions["n_static"], width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.timestep = TimeStepEncoder(dimensions["n_values"], dimensions["n_indicators"], width)
        for module in (self.static, self.timestep):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)
        self.trunk = nn.Sequential(*(MambaBlock(width, n_layers, i) for i in range(n_layers)), RMSNorm(width))
        self.tte_heads = CompactTTEHeads(dimensions["n_targets"], width, tte_hidden, horizon)

    def forward(self, static, values, indicators, valid, anchors, anchor_valid):
        tokens = torch.cat((self.static(static)[:, None], self.timestep(values, indicators)), 1)
        sequence = self.trunk(tokens)[:, 1:]
        rows = torch.arange(len(values), device=values.device)[:, None]
        return sequence[rows, anchors][:, :, None] * anchor_valid[..., None, None]

    def pretrain(self, batch):
        states = self(batch["static"], batch["history"], batch["history_indicators"],
                      batch["history_valid"], batch["anchors"], batch["anchor_valid"]).squeeze(-2)
        return self.tte_heads(states, batch["threshold"])
