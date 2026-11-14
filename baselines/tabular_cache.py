import json
import numpy as np
import torch
from baselines.meds_tab import aggregate_admission, feature_count
from training.cache import load_cache
from utils.labels import eep_labels


def export(grid, config, directory):
    tasks = config["downstream"]["horizons"]
    context = config["finetuning"]["min_context_hours"]
    metadata = dict(data=grid.fingerprint, schema=grid.schema_key, tasks=list(tasks.items()), context=context)
    # JSON normalizes tuples to lists, so compare its canonical serialization.
    metadata = json.loads(json.dumps(metadata))
    if (directory / "complete.json").exists():
        if json.loads((directory / "complete.json").read_text()) != metadata:
            raise ValueError("Tabular cache schema/data mismatch")
        return load_cache(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Incomplete tabular cache; remove explicitly before rebuilding")
    directory.mkdir(parents=True, exist_ok=True)
    rows = int(np.maximum(grid.lengths - context, 0).sum())
    if not rows:
        raise ValueError("No eligible tabular anchors")
    d = grid.dimensions
    width = feature_count(d["n_values"], d["n_indicators"], d["n_static"])
    specs = {"states": ((rows, width), np.float32), "labels": ((rows, len(tasks)), np.uint8),
             "valid": ((rows, len(tasks)), np.bool_), "admission": ((rows,), np.int64), "anchor": ((rows,), np.int64)}
    arrays = {key: np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
              for key, (shape, dtype) in specs.items()}
    print(f"{grid.split}: hourly float32 values/boolean indicators → {rows} × {width} float32 tabular rows", flush=True)
    cursor = 0
    for i in grid.eligible(context):
        source = grid[i]
        anchors = torch.arange(context-1, len(source["values"])-1)
        labels, valid = eep_labels(source["events"], anchors, tasks)
        selection = slice(cursor, cursor+len(anchors))
        arrays["states"][selection] = aggregate_admission(source["values"].numpy(), source["indicators"].numpy(),
                                                         source["static"].numpy(), anchors.numpy())
        arrays["labels"][selection], arrays["valid"][selection] = labels.numpy(), valid.numpy()
        arrays["admission"][selection], arrays["anchor"][selection] = i, anchors.numpy()
        cursor += len(anchors)
    if cursor != rows:
        raise ValueError("Tabular row count mismatch")
    for array in arrays.values():
        array.flush()
    (directory / "complete.json").write_text(json.dumps(metadata, indent=2))
    return load_cache(directory)
