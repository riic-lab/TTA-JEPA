import argparse
import io

import torch
from torch import nn

from model.embeddings import TimeStepEncoder
from model.heads import PrefixTaskHeads
from model.layers import ProjectionMLP
from model.transformer import ModernTransformer
from training.ema import EMATarget


class SmokeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep = TimeStepEncoder(3, 2, 16)
        self.transformer = ModernTransformer(16, 1, 4, 2)
        self.projection = ProjectionMLP(16, 16, 8)

    def forward(self, values, indicators, valid):
        states = self.transformer(self.timestep(values, indicators), valid)
        return self.projection(states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    # Synthetic integration check, not a model objective or scientific experiment.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    encoder = SmokeEncoder()
    target = EMATarget(encoder)
    values = torch.randn(2, 6, 3)
    indicators = torch.ones(2, 6, 2)
    valid = torch.ones(2, 6, dtype=torch.bool)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    prediction = encoder(values, indicators, valid)
    reference = target(values + 0.1, indicators, valid)
    loss = (prediction - reference).square().mean()
    loss.backward()
    nn.utils.clip_grad_norm_(encoder.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    target.update(encoder)
    assert not reference.requires_grad

    encoder.requires_grad_(False).eval()
    encoder.zero_grad(set_to_none=True)
    heads = PrefixTaskHeads(8, {"short": 2, "long": 6}, hidden_dim=16)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3)
    with torch.no_grad():
        cached = encoder(values, indicators, valid)
    logits = heads(cached)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    nn.utils.clip_grad_norm_(heads.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert all(p.grad is None for p in encoder.parameters())

    checkpoint = io.BytesIO()
    torch.save(heads.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = PrefixTaskHeads(8, {"short": 2, "long": 6}, hidden_dim=16)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(heads(cached), restored(cached))
    print("PASS: encoder step, EMA update, frozen-head step, checkpoint reload")


if __name__ == "__main__":
    main()
