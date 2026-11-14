import hashlib
import json

import numpy as np
import torch

from utils.labels import eep_labels


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(directory):
    if not (directory / "complete.json").exists():
        raise ValueError("Incomplete representation cache")
    return {name: np.load(directory / f"{name}.npy", mmap_mode="r")
            for name in ("states", "labels", "valid", "admission", "anchor")}


@torch.no_grad()
def export(grid, encoder, config, directory, checkpoint_hash, device):
    cfg, tasks = config["finetuning"], config["downstream"]["horizons"]
    context = cfg["min_context_hours"]
    metadata = dict(data=grid.fingerprint, checkpoint=checkpoint_hash, context=context,
                    tasks=tasks, task_order=list(tasks), model=config["model"], dtype="float16")
    if (directory / "complete.json").exists():
        if json.loads((directory / "complete.json").read_text()) != metadata:
            raise ValueError("Representation cache configuration mismatch")
        return load_cache(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Incomplete cache exists; preserve/remove it explicitly before rebuilding")
    directory.mkdir(parents=True, exist_ok=True)
    rows = int(np.maximum(grid.lengths - context, 0).sum())
    if not rows:
        raise ValueError("No eligible downstream anchors")
    tokens = getattr(encoder, "n_tokens", config["model"].get("student", {}).get("horizon", 1))
    shape = (rows, tokens, encoder.width)
    specifications = {"states": (shape, np.float16), "labels": ((rows, len(tasks)), np.uint8),
                      "valid": ((rows, len(tasks)), np.bool_), "admission": ((rows,), np.int64),
                      "anchor": ((rows,), np.int64)}
    arrays = {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (shape, dtype) in specifications.items()}
    print(f"{grid.split}: exporting {rows} anchors; raw queries {specifications['states']}", flush=True)
    cursor = 0
    encoder.eval()
    for i in grid.eligible(context):
        source = grid[i]
        all_anchors = torch.arange(context - 1, len(source["values"]) - 1)
        for start in range(0, len(all_anchors), cfg["anchor_chunk_size"]):
            anchors = all_anchors[start:start + cfg["anchor_chunk_size"]]
            end = int(anchors.max()) + 1
            state = encoder(source["static"][None].to(device), source["values"][None, :end].to(device),
                            source["indicators"][None, :end].to(device), torch.ones(1, end, dtype=torch.bool, device=device),
                            anchors[None].to(device), torch.ones(1, len(anchors), dtype=torch.bool, device=device))[0]
            if not torch.isfinite(state).all():
                raise ValueError("Nonfinite representations; cache left incomplete")
            labels, valid = eep_labels(source["events"], anchors, tasks)
            selection = slice(cursor, cursor + len(anchors))
            arrays["states"][selection] = state.to(device="cpu", dtype=torch.float16).numpy()
            arrays["labels"][selection] = labels.numpy().astype(np.uint8)
            arrays["valid"][selection] = valid.numpy()
            arrays["admission"][selection], arrays["anchor"][selection] = i, anchors.numpy()
            cursor += len(anchors)
        if i % 100 == 0:
            print(f"{grid.split} export: {cursor}/{rows} anchors", flush=True)
    if cursor != rows:
        raise ValueError("Export row count mismatch")
    for array in arrays.values():
        array.flush()
    (directory / "complete.json").write_text(json.dumps(metadata, indent=2))
    return load_cache(directory)
