import json
import math
from contextlib import nullcontext
from pathlib import Path
import torch


def autocast(device, config):
    if torch.device(device).type == "cuda" and config.get("precision") == "bfloat16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def optimizer_for(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.ndim > 1], "weight_decay": config["weight_decay"]},
              {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=config["lr"], betas=tuple(config["betas"]), eps=1e-8)
def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def save_checkpoint(path, **state):
    def check(value):
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite checkpoint refused")
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
    check(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def log(path, **record):
    print(json.dumps(record), flush=True)
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record) + "\n")


class Plateau:
    def __init__(self, optimizer, config, maximize=False):
        self.optimizer, self.config = optimizer, config
        self.sign = -1 if maximize else 1
        self.best, self.stale, self.reductions = math.inf, 0, 0

    def step(self, score):
        score *= self.sign
        if not math.isfinite(score):
            raise ValueError("Validation selection score is undefined")
        improved = score < self.best
        if improved:
            self.best, self.stale = score, 0
        else:
            self.stale += 1
        if self.stale >= self.config["patience"]:
            for group in self.optimizer.param_groups:
                group["lr"] = max(group["lr"] * self.config["factor"],
                                  self.config["lr"] * self.config["min_lr_scale"])
            self.reductions += 1
            self.stale = 0
        return improved, self.reductions >= self.config["stop_after_reductions"]

    def state_dict(self):
        return dict(best=self.best, stale=self.stale, reductions=self.reductions)

    def load_state_dict(self, state):
        self.best, self.stale, self.reductions = state["best"], state["stale"], state["reductions"]
