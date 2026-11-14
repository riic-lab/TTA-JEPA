import torch


def admission_statistics(cells, valid, lengths, los_cap, reduce_dims):
    """Additive per-target statistics; sum microbatches before division."""
    if cells.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Cells and boolean validity must have identical shapes")
    if lengths.shape != (cells.size(0),) or (lengths <= 0).any() or los_cap <= 0:
        raise ValueError("Positive admission lengths and LOS cap are required")
    weight = valid.to(cells.dtype)
    count = weight.sum(dim=reduce_dims)
    average = (cells * weight).sum(dim=reduce_dims) / count.clamp_min(1)
    correction = (lengths.float().clamp_max(los_cap) / lengths).sqrt()
    correction = correction.reshape((cells.size(0),) + (1,) * (average.ndim - 1))
    support = count > 0
    return (average * support * correction).sum(0), (support * correction).sum(0)


def supported_mean(numerator, denominator):
    supported = denominator > 0
    return (numerator / denominator.clamp_min(1e-12) * supported).sum() / supported.sum().clamp_min(1)
