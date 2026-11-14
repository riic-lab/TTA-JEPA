import numpy as np
import torch

from evaluation.metrics import metrics
from evaluation.predict import predict
from model.factory import build_model, build_downstream
from .cache import checkpoint_digest, export
from .runtime import Plateau, log, save_checkpoint, autocast


def restore_pretrained(path, config, dimensions, schema_key, device, schema):
    state = torch.load(path, map_location=device, weights_only=False)
    if config.get("method") == "icarefm" and state.get("stage") != "tte_training":
        raise ValueError("ICareFM EEP fitting requires the intermediate TTE-trained checkpoint")
    if state["config"]["model"] != config["model"] or state["dimensions"] != dimensions:
        raise ValueError("Checkpoint architecture or feature dimensions differ")
    if state["schema_key"] != schema_key:
        raise ValueError("Checkpoint feature/target ordering differs")
    model = build_model(config, dimensions, schema).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model


def balance(cache, n_admissions):
    counts = np.zeros((n_admissions, cache["valid"].shape[1]), dtype=np.int64)
    for k in range(counts.shape[1]):
        counts[:, k] = np.bincount(cache["admission"], weights=cache["valid"][:, k], minlength=n_admissions)
    return counts, np.maximum((counts > 0).sum(0), 1)


def finetune(train, val, config, out, source, device, resume=False):
    cfg, tasks = config["finetuning"], config["downstream"]["horizons"]
    pretrained = restore_pretrained(source, config, train.dimensions, train.schema_key, device, train.schema)
    model = build_downstream(pretrained, config).to(device)
    digest = checkpoint_digest(source)
    caches = {grid.split: export(grid, model.representation.encoder, config,
                                  out / "cache" / grid.split, digest, device) for grid in (train, val)}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    schedule = Plateau(optimizer, cfg, maximize=True)
    start_epoch = 1
    if (out / "last.pt").exists():
        if not resume:
            raise FileExistsError("Fine-tuning exists; use --resume or a new output directory")
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if state["config"] != config or state["data"] != [train.fingerprint, val.fingerprint] or state["source"] != digest:
            raise ValueError("Resume configuration, source checkpoint or data mismatch")
        if list(state["config"]["downstream"]["horizons"]) != list(tasks):
            raise ValueError("Resume task order differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        start_epoch = state["epoch"] + 1
    cache = caches["train"]
    counts, supported = balance(cache, len(train))
    n_rows = len(cache["states"])
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if schedule.reductions >= cfg["stop_after_reductions"]:
            break
        torch.manual_seed(config["seed"] + epoch)
        starts = np.arange(0, n_rows, cfg["batch_rows"])
        np.random.default_rng(config["seed"] + epoch).shuffle(starts)
        model.train()
        total = 0.0
        for start in starts:
            rows = slice(start, min(start + cfg["batch_rows"], n_rows))
            states = torch.from_numpy(np.array(cache["states"][rows], dtype=np.float32)).to(device)
            labels = torch.from_numpy(np.array(cache["labels"][rows], dtype=np.float32)).to(device)
            weight = cache["valid"][rows] / np.maximum(counts[cache["admission"][rows]], 1) / supported / len(tasks)
            weight = torch.from_numpy(weight.astype(np.float32)).to(device)
            with autocast(device, cfg):
                logits = model.forward_cached(states)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none") * weight).sum()
            loss = loss * (n_rows / len(states))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        improved, stop, scores = False, False, {}
        if epoch % cfg["val_interval"] == 0:
            probabilities = predict(model, caches["val"], cfg["predict_batch_rows"], device)
            scores = metrics(probabilities, caches["val"]["labels"], caches["val"]["valid"], caches["val"]["admission"], tasks)
            if scores["mean"]["auroc"] is None:
                raise ValueError("No defined validation AUROC; cannot select checkpoint")
            improved, stop = schedule.step(scores["mean"]["auroc"])
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), schedule=schedule.state_dict(),
                     epoch=epoch, config=config, dimensions=train.dimensions, source=digest,
                     data=[train.fingerprint, val.fingerprint], schema_key=train.schema_key)
        save_checkpoint(out / "last.pt", **state)
        if improved:
            save_checkpoint(out / "best.pt", **state)
        log(out / "history.jsonl", phase="finetuning", epoch=epoch, loss=total/len(starts),
            validation=scores, lr=optimizer.param_groups[0]["lr"], reductions=schedule.reductions)
        if stop:
            break
    if not (out / "best.pt").exists():
        raise RuntimeError("No validated downstream checkpoint")
    return out / "best.pt"
