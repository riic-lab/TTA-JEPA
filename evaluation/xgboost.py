import json
import numpy as np
import xgboost as xgb
from baselines.tabular_cache import export
from evaluation.metrics import metrics, save_metrics, write_predictions


def evaluate(grid, config, out, checkpoint, device=None):
    metadata = json.loads((checkpoint / "complete.json").read_text())
    if metadata["schema"] != grid.schema_key or metadata["config"] != config:
        raise ValueError("XGBoost evaluation schema/configuration mismatch")
    cache = export(grid, config, out / "cache" / grid.split)
    probability = np.empty(cache["labels"].shape, np.float32)
    chunk = config["xgboost"]["chunk_rows"]
    tasks = config["downstream"]["horizons"]
    for k, task in enumerate(tasks):
        model = xgb.Booster(model_file=checkpoint / f"{task}.json")
        model.set_param({"nthread": config["cpu_threads"]})
        for start in range(0, len(probability), chunk):
            stop = min(start+chunk, len(probability))
            probability[start:stop, k] = model.inplace_predict(np.asarray(cache["states"][start:stop]))
    scores = metrics(probability, cache["labels"], cache["valid"], cache["admission"], tasks)
    save_metrics(out / f"{grid.split}_metrics.json", scores)
    write_predictions(out / f"{grid.split}_predictions.parquet", probability, cache, tasks, grid.split, grid)
    print(scores["mean"], flush=True)
