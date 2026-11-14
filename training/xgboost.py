import json
import tempfile
from pathlib import Path
import numpy as np
import xgboost as xgb
from baselines.tabular_cache import export
from baselines.xgboost_data import TaskIterator
from training.runtime import log


class Progress(xgb.callback.TrainingCallback):
    def __init__(self, path, task):
        self.path, self.task = path, task

    def after_iteration(self, model, epoch, evals_log):
        if epoch % 10 == 0:
            log(self.path, task=self.task, round=epoch+1, validation_auc=evals_log["val"]["auc"][-1])
        return False


def fit(train, val, config, out, resume=False):
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(config=config, schema=train.schema_key, data=[train.fingerprint, val.fingerprint])
    manifest = out / "training.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError("XGBoost run exists; use --resume")
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("XGBoost resume configuration/data mismatch")
    manifest.write_text(json.dumps(identity, indent=2))
    caches = {g.split: export(g, config, out / "cache" / g.split) for g in (train, val)}
    params = dict(config["xgboost"])
    rounds, patience, chunk = (params.pop(key) for key in ("num_boost_round", "early_stopping_rounds", "chunk_rows"))
    params.update(seed=config["seed"], nthread=config["cpu_threads"])
    for k, task in enumerate(config["downstream"]["horizons"]):
        path = out / f"{task}.json"
        if resume and path.exists():
            continue
        for split, cache in caches.items():
            labels = cache["labels"][cache["valid"][:, k], k]
            if np.unique(labels).size != 2:
                raise ValueError(f"{task}: {split} requires both classes for validation-AUC tree selection")
        with tempfile.TemporaryDirectory(prefix=f"{task}-", dir=out) as temporary:
            it_train = TaskIterator(caches["train"], k, chunk, Path(temporary)/"train")
            it_val = TaskIterator(caches["val"], k, chunk, Path(temporary)/"val")
            training = xgb.ExtMemQuantileDMatrix(it_train, max_bin=params["max_bin"], nthread=config["cpu_threads"])
            validation = xgb.ExtMemQuantileDMatrix(it_val, ref=training, max_bin=params["max_bin"], nthread=config["cpu_threads"])
            model = xgb.train(params, training, rounds, evals=[(validation, "val")], early_stopping_rounds=patience,
                              verbose_eval=False, callbacks=[Progress(out / "history.jsonl", task)])
            model = model[:model.best_iteration+1]
            temporary_model = out / f"{task}.tmp.json"
            model.save_model(temporary_model)
            temporary_model.replace(path)
            del training, validation, it_train, it_val
    (out / "complete.json").write_text(json.dumps(identity, indent=2))
    return out
