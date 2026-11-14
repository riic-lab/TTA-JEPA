import torch
from torch import nn
from torch.nn import functional as F


class HEPALoss(nn.Module):
    """L1 alignment plus predictor-side sliced Epps–Pulley regularization."""

    def __init__(self, config, seed):
        super().__init__()
        self.cfg, self.seed, self.draw = config, seed, 0
        points = torch.linspace(0, 3, config["sigreg_points"])
        spacing = 3 / (len(points) - 1)
        weight = torch.full_like(points, 2 * spacing)
        weight[[0, -1]] = spacing
        self.register_buffer("points", points)
        self.register_buffer("reference", (-points.square() / 2).exp())
        self.register_buffer("weight", weight * self.reference)

    def reset(self):
        self.draw = 0

    def forward(self, prediction, target):
        values = prediction.float()
        if len(values) < self.cfg["sigreg_min_samples"]:
            raise ValueError("HEPA microbatch too small for SIGReg; increase batch/token capacity")
        rng = torch.Generator(device=values.device).manual_seed(self.seed + self.draw)
        total = values.new_zeros(())
        for start in range(0, self.cfg["sigreg_slices"], self.cfg["sigreg_chunk"]):
            width = min(self.cfg["sigreg_chunk"], self.cfg["sigreg_slices"] - start)
            directions = F.normalize(torch.randn(values.size(-1), width, generator=rng, device=values.device), dim=0)
            phase = (values @ directions)[..., None] * self.points
            error = (phase.cos().mean(0) - self.reference).square() + phase.sin().mean(0).square()
            total = total + (error @ self.weight).sum()
        self.draw += 1
        regularization = total / self.cfg["sigreg_slices"] * self.cfg["sigreg_reference_batch"]
        alignment = F.l1_loss(values, target.float())
        alpha = self.cfg["sigreg_alpha"]
        return (1-alpha)*alignment + alpha*regularization
