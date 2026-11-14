import numpy as np
import torch

from model.factory import build_model, build_downstream
from training.cache import checkpoint_digest, export
from .metrics import metrics, save_metrics, write_predictions


@torch.no_grad()
def predict(model, cache, batch_rows, device):
    result = np.empty(cache["labels"].shape, dtype=np.float32)
    model.eval()
    for start in range(0, len(result), batch_rows):
        end = min(start + batch_rows, len(result))
        states = torch.from_numpy(np.array(cache["states"][start:end], dtype=np.float32)).to(device)
        result[start:end] = model.forward_cached(states).sigmoid().cpu().numpy()
    return result


def evaluate(grid, config, out, checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["config"]["model"] != config["model"] or state["dimensions"] != grid.dimensions:
        raise ValueError("Evaluation checkpoint does not match data/configuration")
    if state["schema_key"] != grid.schema_key:
        raise ValueError("Evaluation feature/target ordering differs")
    if state["config"]["downstream"] != config["downstream"]:
        raise ValueError("Evaluation task layout differs from trained heads")
    if list(state["config"]["downstream"]["horizons"]) != list(config["downstream"]["horizons"]):
        raise ValueError("Evaluation task order differs from trained heads")
    model = build_downstream(build_model(config, grid.dimensions, grid.schema), config).to(device)
    model.load_state_dict(state["model"], strict=True)
    cache = export(grid, model.representation.encoder, config, out / "cache" / grid.split,
                   checkpoint_digest(checkpoint), device)
    probability = predict(model, cache, config["finetuning"]["predict_batch_rows"], device)
    tasks = config["downstream"]["horizons"]
    scores = metrics(probability, cache["labels"], cache["valid"], cache["admission"], tasks)
    save_metrics(out / f"{grid.split}_metrics.json", scores)
    write_predictions(out / f"{grid.split}_predictions.parquet", probability, cache, tasks, grid.split, grid)
    print(scores["mean"], flush=True)
