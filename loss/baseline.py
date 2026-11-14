import torch
from torch.nn import functional as F
from loss.next_state import NextStateLoss
from loss.reduction import admission_statistics, supported_mean
from loss.hepa import HEPALoss


class BaselineObjective:
    def __init__(self, kind, grid, cfg, seed, cap, device):
        self.kind, self.cap = kind, cap
        self.next_state = NextStateLoss(grid.schema, cap) if kind == "next_state" else None
        self.hepa = HEPALoss(cfg, seed).to(device) if kind == "hepa" else None

    def denominator(self, batch):
        if self.kind == "next_state":
            return self.next_state.denominators(batch)
        if self.kind == "tte":
            valid = batch["tte_valid"]
            return admission_statistics(torch.zeros_like(valid, dtype=torch.float32), valid,
                                         batch["lengths"], self.cap, (1, 3))[1]
        return torch.tensor(float(len(batch["static"])))

    def statistics(self, model, batch):
        if self.kind == "hepa":
            loss = self.hepa(*model.pretrain(batch))
            return loss * len(batch["static"]), loss.new_tensor(len(batch["static"]))
        if self.kind == "next_state":
            return self.next_state.statistics(model.pretrain(batch), batch)
        logits = model.supervised(batch) if hasattr(model, "supervised") else model.pretrain(batch)
        cells = F.binary_cross_entropy_with_logits(logits.float(), batch["tte_labels"], reduction="none")
        return admission_statistics(cells, batch["tte_valid"], batch["lengths"], self.cap, (1, 3))

    reduce = staticmethod(supported_mean)
