from torch import nn
from torch.nn import functional as F

from .reduction import admission_statistics, supported_mean


class TTEJEPALoss(nn.Module):
    def __init__(self, los_cap: float):
        super().__init__()
        self.los_cap = los_cap

    def statistics(self, output, batch):
        prediction, target = output["prediction"], output["target"].detach()
        if prediction.shape != target.shape:
            raise ValueError("Student and target shapes must agree")
        cells = (F.normalize(prediction.float(), dim=-1)
                 - F.normalize(target.float(), dim=-1)).square().sum(-1)
        # Intentionally all query horizons of real anchors, not future_valid.
        query_valid = batch["anchor_valid"][..., None].expand_as(cells)
        jepa_num, jepa_den = admission_statistics(
            cells, query_valid, batch["lengths"], self.los_cap, (1, 2))
        tte_cells = F.binary_cross_entropy_with_logits(
            output["tte_logits"].float(), batch["tte_labels"].float(), reduction="none")
        tte_num, tte_den = admission_statistics(
            tte_cells, batch["tte_valid"], batch["lengths"], self.los_cap, (1, 3))
        return {"jepa_num": jepa_num, "jepa_den": jepa_den,
                "tte_num": tte_num, "tte_den": tte_den}

    @staticmethod
    def reduce(statistics):
        jepa = statistics["jepa_num"].sum() / statistics["jepa_den"].sum().clamp_min(1)
        tte = supported_mean(statistics["tte_num"], statistics["tte_den"])
        return {"loss": jepa + tte, "jepa": jepa, "tte": tte}

    def forward(self, output, batch):
        return self.reduce(self.statistics(output, batch))
