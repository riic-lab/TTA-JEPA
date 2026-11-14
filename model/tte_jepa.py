from copy import deepcopy

from torch import nn

from training.ema import EMATarget
from .heads import PrefixTaskHeads
from .history import HistoryQueryEncoder, HistoryRepresentation
from .teacher import FutureTeacher
from .tte_heads import TTEHeads


class TTEJEPA(nn.Module):
    def __init__(self, n_static, n_values, n_indicators, n_targets, *,
                 student, teacher, tte_head, ema_momentum=0.996):
        super().__init__()
        if student["horizon"] != teacher["horizon"]:
            raise ValueError("Student and teacher horizons must match")
        self.student = HistoryRepresentation(
            HistoryQueryEncoder(n_static, n_values, n_indicators, **student), teacher["width"])
        self.live_teacher = FutureTeacher(n_values, n_indicators, **teacher)
        self.ema_teacher = EMATarget(self.live_teacher, ema_momentum)
        self.tte_heads = TTEHeads(n_targets, width=teacher["width"], **tte_head)

    def forward(self, batch):
        future = (batch["future"], batch["future_indicators"], batch["future_valid"])
        live = self.live_teacher(*future)
        return {
            "prediction": self.student(batch),
            "target": self.ema_teacher(*future),
            "tte_logits": self.tte_heads(live, batch["threshold"]),
        }

    def update_target(self):
        self.ema_teacher.update(self.live_teacher)


class TTEJEPADownstream(nn.Module):
    """Copy the pretrained student; fit its projection and fresh EEP heads."""

    def __init__(self, pretrained: TTEJEPA, horizons, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.representation = deepcopy(pretrained.student)
        self.representation.encoder.requires_grad_(False).eval()
        self.heads = PrefixTaskHeads(self.representation.output.out_features,
                                     horizons, hidden_dim, dropout)

    def train(self, mode=True):
        super().train(mode)
        self.representation.encoder.eval()
        return self

    def forward_cached(self, queries):
        return self.heads(self.representation.project(queries.detach()))

    def forward(self, batch):
        return self.heads(self.representation(batch))
