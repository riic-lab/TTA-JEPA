import zlib

import numpy as np
import torch

from .labels import tte_labels


def generator(seed, epoch, identity, draw):
    identity = zlib.crc32(str(identity).encode())
    state = np.random.SeedSequence([seed, epoch, identity, draw]).generate_state(1, dtype=np.uint64)
    return torch.Generator().manual_seed(int(state[0] % np.uint64(torch.iinfo(torch.int64).max)))


def corrupt(values, indicators, valid, grid, rng, config):
    values, indicators = values.clone(), indicators.clone()
    targets = set(grid.target_indices.tolist())
    ordinary = torch.tensor([i for i, x in enumerate(grid.continuous.tolist()) if x not in targets], dtype=torch.long)
    for a in range(len(values)):
        hours = int(valid[a].sum())
        span = min(round(hours * config["temporal_mask_rate"]), max(0, hours - 1))
        if span:
            start = int(torch.randint(hours - span + 1, (), generator=rng))
            values[a, start:start + span] = 0
            indicators[a, start:start + span] = 0
        dropped = ordinary[torch.rand(len(ordinary), generator=rng) < config["feature_mask_rate"]]
        target_drop = grid.target_indices[torch.rand(len(grid.target_indices), generator=rng) < config["target_feature_dropout"]]
        values[a, :hours, grid.continuous[dropped]] = 0
        indicators[a, :hours, dropped] = 0
        values[a, :hours, target_drop] = 0
        for target in target_drop:
            matches = (grid.continuous == target).nonzero().flatten()
            if len(matches) != 1:
                raise ValueError("Target must have exactly one observation indicator")
            indicators[a, :hours, matches] = 0
    return values, indicators


def sample(grid, index, config, seed, epoch, draw=0):
    source = grid[index]
    length = len(source["values"])
    lower = config["min_context_hours"] - 1
    eligible = length - 1 - lower
    count = min(config["anchors_per_admission"], eligible)
    if count == eligible:
        anchors = torch.arange(lower, length - 1)
    else:
        edges = torch.linspace(0, eligible, count + 1).floor().long()
        widths = edges[1:] - edges[:-1]
        offsets = (torch.rand(count, generator=generator(seed, epoch, grid.ids[index], draw)) * widths).floor().long()
        anchors = lower + edges[:-1] + torch.minimum(offsets, widths - 1)
    indices = anchors[:, None] + torch.arange(1, 49)[None]
    valid = indices < length
    safe = indices.clamp_max(length - 1)
    future = source["values"][safe] * valid[..., None]
    indicators = source["indicators"][safe] & valid[..., None]
    future, indicators = corrupt(future, indicators, valid, grid,
                                  generator(seed, epoch, grid.ids[index], 10_000_000 + draw), config)
    rng = generator(seed, epoch, grid.ids[index], 20_000_000 + draw)
    threshold = torch.empty(count, len(grid.targets)).uniform_(-4, 4, generator=rng)
    labels, label_valid = tte_labels(source["trace"], anchors, threshold)
    end = int(anchors.max()) + 1
    return dict(static=source["static"], history=source["values"][:end], history_indicators=source["indicators"][:end],
                anchors=anchors, future=future, future_indicators=indicators, future_valid=valid,
                threshold=threshold, tte_labels=labels, tte_valid=label_valid,
                lengths=torch.tensor(length))


def collate(items):
    result = {}
    for key in items[0]:
        tensors = [item[key] for item in items]
        result[key] = torch.stack(tensors) if key in ("static", "lengths") else torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    result["history_valid"] = torch.arange(result["history"].size(1))[None] < torch.tensor([len(x["history"]) for x in items])[:, None]
    result["anchor_valid"] = torch.arange(result["anchors"].size(1))[None] < torch.tensor([len(x["anchors"]) for x in items])[:, None]
    return result
