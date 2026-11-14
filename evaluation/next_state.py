import torch
from utils.sampling import generator
from loss.next_state import NextStateLoss


@torch.no_grad()
def validate_rollout(model, grid, indices, cap, seed, device):
    objective = NextStateLoss(grid.schema, cap)
    numerator = denominator = None
    for index in indices:
        source = grid[index]
        length = len(source["values"])
        cut = int(torch.randint(1, length, (), generator=generator(seed, 0, grid.ids[index], 0)))
        horizon = min(48, length-cut)
        prediction = model.rollout(source["static"][None].to(device), source["values"][None, :cut].to(device),
                                   source["indicators"][None, :cut].to(device),
                                   torch.tensor([[cut-1]], device=device), torch.ones(1, 1, dtype=torch.bool, device=device))[:, 0, :horizon]
        batch = dict(history=source["values"][None, cut-1:cut+horizon].to(device),
                     history_indicators=source["indicators"][None, cut-1:cut+horizon].to(device),
                     history_valid=torch.ones(1, horizon+1, dtype=torch.bool, device=device),
                     lengths=torch.tensor([length], device=device))
        n, d = objective.statistics(prediction, batch)
        numerator = n if numerator is None else numerator + n
        denominator = d if denominator is None else denominator + d
    return float(objective.reduce(numerator, denominator))
