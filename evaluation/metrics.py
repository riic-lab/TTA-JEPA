import json

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(probability, labels, valid, admission, tasks):
    result = {}
    for k, task in enumerate(tasks):
        keep = valid[:, k].astype(bool)
        ids, y, p = admission[keep], labels[keep, k], probability[keep, k]
        if not np.isfinite(p).all():
            raise ValueError("Nonfinite predictions")
        weight = 1 / np.bincount(ids)[ids] if len(ids) else np.empty(0)
        defined = len(np.unique(y)) == 2
        result[task] = dict(auroc=float(roc_auc_score(y, p, sample_weight=weight)) if defined else None,
                            auprc=float(average_precision_score(y, p, sample_weight=weight)) if defined else None,
                            n_valid=int(keep.sum()), n_positive=int(y.sum()))
    result["mean"] = {}
    for metric in ("auroc", "auprc"):
        scores = [result[t][metric] for t in tasks if result[t][metric] is not None]
        result["mean"][metric] = float(np.mean(scores)) if scores else None
    return result


def write_predictions(path, probability, cache, tasks, split, grid):
    columns = {"split": [split] * len(probability), "admission_index": cache["admission"],
               "anchor": cache["anchor"]}
    for k, task in enumerate(tasks):
        for name, array in (("probability", probability), ("label", cache["labels"]), ("valid", cache["valid"])):
            columns[f"{task}/{name}"] = array[:, k]
    pl.DataFrame(columns).write_parquet(path)
    pl.DataFrame({"admission_index": np.arange(len(grid)), "admissionid": [str(x) for x in grid.ids]}).write_parquet(
        path.with_name(f"{split}_admissions.parquet"))


def save_metrics(path, scores):
    path.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
