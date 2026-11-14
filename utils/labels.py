import torch


def tte_labels(trace, anchors, threshold, horizon=48):
    indices = anchors[:, None] + torch.arange(1, horizon + 1)[None]
    in_bounds = indices < len(trace)
    future = trace[indices.clamp_max(len(trace) - 1)].transpose(1, 2)
    future = torch.where(in_bounds[:, None], future, float("nan"))
    crossing = ((threshold[..., None] > 0) & (future > threshold[..., None])) | (
        (threshold[..., None] < 0) & (future < threshold[..., None]))
    labels = ((crossing.cumsum(-1) > 0) & in_bounds[:, None]).float()
    # Preserve the window-level observation-support rule for TTE supervision.
    valid = (~future.isnan()).any(-1, keepdim=True) & in_bounds[:, None]
    return labels, valid


def eep_labels(events, anchors, horizons):
    names = tuple(horizons)
    state_names = ("circulatory_failure", "respiratory_failure", "kidney_failure", "liver_failure",
                   "hyperglycemia", "sepsis3", "decompensation")
    if set(names) != set(state_names):
        raise ValueError("Prepared clinical states require the documented seven tasks")
    labels = torch.zeros(len(anchors), len(names))
    valid = torch.zeros_like(labels, dtype=torch.bool)
    for k, (name, horizon) in enumerate(horizons.items()):
        state_column = state_names.index(name)
        remaining = len(events) - 1 - anchors
        if name == "decompensation":
            labels[:, k] = ((events[-1, state_column] == 1) & (remaining > 0) & (remaining <= horizon)).float()
            valid[:, k] = remaining > 0
            continue
        indices = anchors[:, None] + torch.arange(1, horizon + 1)[None]
        future = events[indices.clamp_max(len(events) - 1), state_column]
        future = torch.where(indices < len(events), future, float("nan"))
        event, stable = (future == 1).any(-1), (future == 0).any(-1)
        labels[:, k] = event.float()
        valid[:, k] = (events[anchors, state_column] != 1) & (event | ((remaining >= horizon) & stable))
    return labels, valid
