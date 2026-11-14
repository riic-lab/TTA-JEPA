import torch
from torch import nn
from torch.nn import functional as F
from loss.reduction import admission_statistics, supported_mean


def feature_groups(schema):
    dynamic = [column for spec in schema.values() if not spec["reconstruction_type"].startswith("static_")
               for column in spec["physical_columns"]]
    continuous = [spec for spec in schema.values() if "presence_mask_column" in spec
                  and not spec["reconstruction_type"].startswith("static_")]
    categorical = [spec for spec in schema.values() if spec["reconstruction_type"] == "categorical"]
    groups = [("continuous", [dynamic.index(spec["physical_columns"][0])], i, None)
              for i, spec in enumerate(continuous)]
    for i, spec in enumerate(categorical):
        missing = [dynamic.index(c) for c in spec["physical_columns"] if c.endswith("__missing")]
        if len(missing) != 1:
            raise ValueError("Categorical features require one explicit missing class")
        real = [dynamic.index(c) for c in spec["physical_columns"] if not c.endswith("__missing")]
        groups.append(("categorical", real, len(continuous) + i, missing[0]))
    return groups


class NextStateLoss(nn.Module):
    """Observed continuous MSE and categorical CE; treatments are inputs only."""

    def __init__(self, schema, cap):
        super().__init__()
        self.groups, self.cap = feature_groups(schema), cap

    def cells(self, prediction, batch):
        target = batch["history"][:, 1:]
        observed = batch["history_indicators"][:, 1:]
        valid = batch["history_valid"][:, 1:]
        losses, masks = [], []
        for kind, columns, mask, _ in self.groups:
            if kind == "continuous":
                error = (prediction[..., columns[0]] - target[..., columns[0]]).square()
            else:
                logits, labels = prediction[..., columns], target[..., columns].argmax(-1)
                error = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
            losses.append(error)
            masks.append(valid & observed[..., mask].bool())
        return torch.stack(losses, -1), torch.stack(masks, -1)

    def statistics(self, prediction, batch):
        error, valid = self.cells(prediction, batch)
        return admission_statistics(error, valid, batch["lengths"], self.cap, (1,))

    def denominators(self, batch):
        return self.statistics(torch.zeros_like(batch["history"][:, 1:]), batch)[1]

    @staticmethod
    def reduce(numerator, denominator):
        return supported_mean(numerator, denominator)
