from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from model.heads import CLSTaskHeads


class FrozenReadout(nn.Module):
    def __init__(self, pretrained, method, horizons, hidden_dim, dropout):
        super().__init__()
        self.encoder = deepcopy(pretrained).requires_grad_(False).eval()
        self.method = method
        if method == "hepa":
            self.predictors = nn.ModuleList([deepcopy(pretrained.predictor) for _ in horizons])
            self.heads = nn.ModuleList([deepcopy(pretrained.event_head) for _ in horizons])
            self.horizons = tuple(horizons.values())
        else:
            self.heads = CLSTaskHeads(pretrained.width, tuple(horizons), hidden_dim, dropout)

    @property
    def representation(self):
        # Same extraction interface as the TTE-JEPA downstream model.
        return SimpleNamespace(encoder=self.encoder)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward_cached(self, states):
        states = states[..., 0, :]
        if self.method == "hepa":
            return torch.stack([head(predictor(torch.cat((states, states.new_full((*states.shape[:-1], 1), h)), -1))).squeeze(-1)
                                for h, predictor, head in zip(self.horizons, self.predictors, self.heads)], -1)
        return self.heads(states)
