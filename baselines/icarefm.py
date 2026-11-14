import torch
from torch import nn
from torch.nn import functional as F
from model.embeddings import TimeStepEncoder, FourierPositionEncoding
from model.vanilla_cache import prefill, decode
from loss.next_state import feature_groups
from model.compact_tte import CompactTTEHeads


class ICareFM(nn.Module):
    def __init__(self, dimensions, schema, width, n_layers, n_heads, horizon, dropout, tte_hidden):
        super().__init__()
        self.width, self.n_tokens, self.horizon = width, 1, horizon
        self.groups = feature_groups(schema)
        self.static = nn.Sequential(nn.Linear(dimensions["n_static"], width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.timestep = TimeStepEncoder(dimensions["n_values"], dimensions["n_indicators"], width)
        self.position = FourierPositionEncoding(width)
        layer = nn.TransformerEncoderLayer(width, n_heads, 4 * width, dropout,
                                            activation=F.silu, batch_first=True, norm_first=False)
        self.trunk = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(width, dimensions["n_values"])
        self.tte_heads = CompactTTEHeads(dimensions["n_targets"], width, tte_hidden, horizon)

    def tokens(self, static, values, indicators):
        content = torch.cat((self.static(static)[:, None], self.timestep(values, indicators)), 1)
        return content + self.position(content.size(1), device=content.device, dtype=content.dtype)[None]

    def pretrain(self, batch):
        tokens = self.tokens(batch["static"], batch["history"], batch["history_indicators"])
        mask = torch.ones(tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool).triu(1)
        states = self.trunk(tokens, mask=mask)[:, 1:-1]
        return self.decoder(states)

    def forward(self, static, values, indicators, valid, anchors, anchor_valid):
        tokens = self.tokens(static, values, indicators)
        mask = torch.ones(tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool).triu(1)
        states = self.trunk(tokens, mask=mask)[:, 1:]
        rows = torch.arange(len(values), device=values.device)[:, None]
        return states[rows, anchors][:, :, None] * anchor_valid[..., None, None]

    def supervised(self, batch):
        states = self(batch["static"], batch["history"], batch["history_indicators"],
                      batch["history_valid"], batch["anchors"], batch["anchor_valid"]).squeeze(-2)
        return self.tte_heads(states, batch["threshold"])

    def feedback(self, raw):
        value = raw.clone()
        for kind, columns, _, missing in self.groups:
            if kind == "categorical":
                winner = raw[..., columns].argmax(-1)
                value[..., columns] = F.one_hot(winner, len(columns)).to(raw.dtype)
                value[..., missing] = 0
        return value

    @torch.no_grad()
    def rollout(self, static, values, indicators, anchors, anchor_valid):
        if self.training:
            raise ValueError("Cached ICareFM rollout requires evaluation mode")
        result = values.new_zeros(*anchors.shape, self.horizon, self.decoder.out_features)
        # Prefill each admission once; fan-out only the requested anchor chunk.
        for b in range(len(values)):
            tokens = self.tokens(static[b:b+1], values[b:b+1], indicators[b:b+1])
            hidden, shared = prefill(self.trunk.layers, tokens)
            for a in anchor_valid[b].nonzero().flatten().tolist():
                anchor = int(anchors[b, a])
                length = anchor + 2  # Includes the static token.
                caches = [(k[..., :length, :], v[..., :length, :]) for k, v in shared]
                current = hidden[:, length-1]
                positions = self.position(length + self.horizon, device=values.device, dtype=values.dtype)
                for h in range(self.horizon):
                    raw = self.decoder(current)
                    result[b, a, h] = raw[0]
                    feedback = self.feedback(raw)[:, None]
                    mask = indicators.new_zeros(1, 1, indicators.size(-1))
                    token = self.timestep(feedback, mask) + positions[length+h][None, None]
                    current = decode(self.trunk.layers, token, caches)[:, 0]
        return result
