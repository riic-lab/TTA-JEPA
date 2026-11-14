import math
import numpy as np
import torch
from utils.sampling import sample, generator


def item(grid, index, draw, epoch, seed, cfg, objective):
    if objective == "tte":
        clean = dict(cfg, temporal_mask_rate=0, feature_mask_rate=0, target_feature_dropout=0)
        return sample(grid, index, clean, seed, epoch, draw)
    source = grid[index]
    result = dict(static=source["static"], history=source["values"],
                  history_indicators=source["indicators"], lengths=torch.tensor(len(source["values"])))
    if objective == "next_state":
        return result
    length, count = len(source["values"]), cfg["anchors_per_admission"]
    lower = cfg["min_context_hours"] - 1
    count = min(count, length - 1 - lower)
    edges = np.linspace(lower, length - 1, count + 1).astype(int)
    rng = generator(seed, epoch, grid.ids[index], draw)
    slot = draw % count
    anchor = int(torch.randint(int(edges[slot]), int(edges[slot+1]), (), generator=rng))
    maximum = min(48, length - anchor - 1)
    delta = min(maximum, max(1, int(math.exp(float(torch.rand((), generator=rng)) * math.log(maximum)))))
    result.update(history=source["values"][:anchor+1], history_indicators=source["indicators"][:anchor+1],
                  future=source["values"][anchor+1:anchor+1+delta],
                  future_indicators=source["indicators"][anchor+1:anchor+1+delta], delta=torch.tensor(float(delta)))
    return result


def collate(items):
    batch = {}
    for key in items[0]:
        values = [x[key] for x in items]
        batch[key] = torch.stack(values) if key in ("static", "lengths", "delta") else torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
    for key in ("history", "future", "anchors"):
        if key in batch:
            name = "anchor_valid" if key == "anchors" else key + "_valid"
            # TTE samples already carry a mask for their [A,H] future tensor.
            if name not in batch:
                batch[name] = torch.arange(batch[key].size(1))[None] < torch.tensor([len(x[key]) for x in items])[:, None]
    return batch


def batches(items, cfg):
    pending, maximum = [], 0
    for value in items:
        length = len(value["history"]) + (len(value["future"]) if "delta" in value else 0)
        if pending and (len(pending) >= cfg["batch_size"] or max(maximum, length) * (len(pending)+1) > cfg["max_batch_tokens"]):
            yield collate(pending)
            pending, maximum = [], 0
        pending.append(value)
        maximum = max(maximum, length)
    if pending:
        yield collate(pending)


def sample_order(grid, cfg, objective, validation=False):
    indices = grid.eligible(cfg["min_context_hours"])
    draws = cfg["validation_draws"] if validation else 1
    if objective == "hepa" and not validation:
        return [(i, d) for i in indices for d in range(min(cfg["anchors_per_admission"], int(grid.lengths[i])-cfg["min_context_hours"]))]
    return [(i, d) for i in indices for d in range(draws if objective != "next_state" else 1)]
