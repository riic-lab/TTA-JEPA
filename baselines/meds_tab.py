"""MEDS-TAB cumulative feature aggregation for hourly ICU grids."""

from __future__ import annotations

import numpy as np
import pandas as pd

VALUE_STATISTICS = ("min", "max", "mean", "median", "last")
INDICATOR_STATISTICS = ("count", "rate")
WINDOWS = ("full", "half")


def feature_count(n_values: int, n_indicators: int, n_static: int) -> int:
    """Return the width emitted by :func:`aggregate_admission`."""
    per_window = n_values * len(VALUE_STATISTICS)
    per_window += n_indicators * len(INDICATOR_STATISTICS)
    return len(WINDOWS) * per_window + n_static


def feature_names(value_names, indicator_names, static_names) -> list[str]:
    """Return names in exactly the same order as the feature matrix."""
    names: list[str] = []
    for window in WINDOWS:
        for statistic in VALUE_STATISTICS:
            names.extend(f"{window}__{statistic}__{name}" for name in value_names)
        for statistic in INDICATOR_STATISTICS:
            names.extend(f"{window}__{statistic}__{name}" for name in indicator_names)
    names.extend(f"static__{name}" for name in static_names)
    return names


def aggregate_admission(
    values: np.ndarray,
    indicators: np.ndarray,
    static: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Map `[T,D]`, `[T,M]`, `[S]`, `[A]` to MEDS-TAB rows `[A,F]`.

    Both windows are causal and inclusive: the full window ends at the anchor,
    while the half window ends at floor(anchor / 2).
    """
    x = np.asarray(values, dtype=np.float32)
    m = np.asarray(indicators, dtype=np.float32)
    static = np.asarray(static, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.int64)
    if x.ndim != 2 or m.ndim != 2 or x.shape[0] != m.shape[0]:
        raise ValueError("values and indicators must have shapes [T,D] and [T,M]")
    if static.ndim != 1 or anchors.ndim != 1:
        raise ValueError("static and anchors must be one-dimensional")
    if anchors.size and (int(anchors.min()) < 0 or int(anchors.max()) >= x.shape[0]):
        raise ValueError("anchor lies outside the admission")

    denominators = np.arange(1, x.shape[0] + 1, dtype=np.float64)[:, None]
    cumulative = {
        "min": np.minimum.accumulate(x, axis=0),
        "max": np.maximum.accumulate(x, axis=0),
        "mean": np.cumsum(x, axis=0, dtype=np.float64) / denominators,
        "median": pd.DataFrame(x, copy=False)
        .expanding(min_periods=1)
        .median()
        .to_numpy(dtype=np.float32, copy=False),
        "last": x,
        "count": np.cumsum(m, axis=0, dtype=np.float64),
        "rate": np.cumsum(m, axis=0, dtype=np.float64) / denominators,
    }
    blocks: list[np.ndarray] = []
    for ends in (anchors, anchors // 2):
        blocks.extend(cumulative[name][ends] for name in VALUE_STATISTICS)
        blocks.extend(cumulative[name][ends] for name in INDICATOR_STATISTICS)
    blocks.append(np.broadcast_to(static, (anchors.size, static.size)))
    output = np.concatenate(blocks, axis=1, dtype=np.float32)
    expected = feature_count(x.shape[1], m.shape[1], static.size)
    if output.shape != (anchors.size, expected):
        raise AssertionError((output.shape, (anchors.size, expected)))
    return output
