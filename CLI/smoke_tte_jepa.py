import argparse
import io
from pathlib import Path

import torch
import yaml

from loss.tte_jepa import TTEJEPALoss
from model.tte_jepa import TTEJEPA, TTEJEPADownstream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/tte_jepa.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    # Small synthetic integration run, not a change to the released defaults.
    config["model"]["student"].update(width=16, n_layers=1, n_heads=4, n_kv_heads=4)
    config["model"]["teacher"].update(width=8, n_heads=2)
    config["model"]["tte_head"]["hidden_dim"] = 16
    config["downstream"]["hidden_dim"] = 16
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = TTEJEPA(2, 3, 2, 2, **config["model"])
    batch = {
        "static": torch.randn(2, 2), "history": torch.randn(2, 8, 3),
        "history_indicators": torch.ones(2, 8, 2),
        "history_valid": torch.tensor([[True]*8, [True]*6 + [False]*2]),
        "anchors": torch.tensor([[3, 6], [4, -1]]),
        "anchor_valid": torch.tensor([[True, True], [True, False]]),
        "future": torch.randn(2, 2, 48, 3), "future_indicators": torch.ones(2, 2, 48, 2),
        "future_valid": torch.ones(2, 2, 48, dtype=torch.bool),
        "threshold": torch.randn(2, 2, 2),
        "tte_labels": torch.zeros(2, 2, 2, 48),
        "tte_valid": torch.ones(2, 2, 2, 48, dtype=torch.bool),
        "lengths": torch.tensor([80, 60]),
    }
    batch["future_valid"][0, 1, 30:] = False
    batch["future_valid"][1, 1] = False
    batch["tte_valid"] &= batch["future_valid"][:, :, None]
    batch["tte_labels"][0, 0, 0, 12:] = 1
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    output = model(batch)
    losses = TTEJEPALoss(los_cap=75)(output, batch)
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    assert model.student.encoder.time_step_encoder.fc1.weight.grad is not None
    assert model.live_teacher.time_step_encoder.fc1.weight.grad is not None
    assert not output["target"].requires_grad
    optimizer.step()
    model.update_target()

    # End-to-end causal check: modifying later future rows cannot affect earlier states.
    with torch.no_grad():
        original = model.live_teacher(batch["future"], batch["future_indicators"], batch["future_valid"])
        changed = batch["future"].clone()
        changed[:, :, 20:] += 10
        altered = model.live_teacher(changed, batch["future_indicators"], batch["future_valid"])
        torch.testing.assert_close(original[:, :, :20], altered[:, :, :20])

    downstream = TTEJEPADownstream(model, **config["downstream"])
    model.zero_grad(set_to_none=True)
    logits = downstream(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert all(p.grad is None for p in downstream.representation.encoder.parameters())
    assert downstream.representation.adapter.net[1].weight.grad is not None
    optimizer = torch.optim.AdamW((p for p in downstream.parameters() if p.requires_grad), lr=1e-3)
    torch.nn.utils.clip_grad_norm_(downstream.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    checkpoint = io.BytesIO()
    torch.save(downstream.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = TTEJEPADownstream(model, **config["downstream"])
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(downstream(batch), restored(batch))
    print("PASS: TTE-JEPA pretraining, independent encoders, causal teacher, EMA, downstream step, reload")
    print({name: round(value.item(), 6) for name, value in losses.items()})


if __name__ == "__main__":
    main()
