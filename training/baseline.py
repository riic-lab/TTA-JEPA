import math
import numpy as np
import torch
from model.factory import build_model
from utils.baseline_sampling import item, batches, sample_order
from loss.baseline import BaselineObjective
from training.runtime import Plateau, log, move, save_checkpoint, optimizer_for
from training.cache import checkpoint_digest
from evaluation.next_state import validate_rollout
from training.hepa_step import step as hepa_step


@torch.no_grad()
def validate(model, grid, cfg, objective, seed, device):
    model.eval()
    order = sample_order(grid, cfg, objective.kind, validation=True)
    if objective.kind == "next_state":
        return validate_rollout(model, grid, [i for i, _ in order], objective.cap, seed, device)
    if objective.hepa:
        objective.hepa.reset()
        predictions, targets = [], []
        for cpu in batches((item(grid, i, draw, 0, seed, cfg, objective.kind) for i, draw in order), cfg):
            p, t = model.pretrain(move(cpu, device))
            predictions.append(p.cpu())
            targets.append(t.cpu())
        prediction, target = torch.cat(predictions), torch.cat(targets)
        groups = max(1, math.ceil(len(prediction) / cfg["effective_batch_size"]))
        parts = torch.tensor_split(torch.arange(len(prediction)), groups)
        return sum(float(objective.hepa(prediction[part].to(device), target[part].to(device))) * len(part)
                   for part in parts) / len(prediction)
    values = (item(grid, i, draw, 0, seed, cfg, objective.kind) for i, draw in order)
    total_n = total_d = None
    for cpu in batches(values, cfg):
        n, d = objective.statistics(model, move(cpu, device))
        total_n = n if total_n is None else total_n + n
        total_d = d if total_d is None else total_d + d
    return float(objective.reduce(total_n, total_d))


def pretrain(train, val, config, out, device, resume=False, source=None, stage="pretraining"):
    cfg = dict(config["pretraining"])
    if stage == "tte_training":
        cfg.update(config["tte_training"], objective="tte")
    model = build_model(config, train.dimensions, train.schema).to(device)
    digest = checkpoint_digest(source) if source is not None else None
    if source is not None:
        state = torch.load(source, map_location=device, weights_only=False)
        if state["config"]["model"] != config["model"] or state["schema_key"] != train.schema_key:
            raise ValueError("Source checkpoint architecture/schema mismatch")
        model.load_state_dict(state["model"])
    cap = float(np.quantile(train.lengths, 0.9))
    objective = BaselineObjective(cfg["objective"], train, cfg, config["seed"], cap, device)
    optimizer = optimizer_for(model, cfg)
    schedule = Plateau(optimizer, cfg)
    out.mkdir(parents=True, exist_ok=True)
    start, step = 1, 0
    if (out / "last.pt").exists():
        if not resume:
            raise FileExistsError("Training exists; use --resume or a new output directory")
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if state["config"] != config or state["data"] != [train.fingerprint, val.fingerprint] or state["source"] != digest:
            raise ValueError("Resume configuration, data or source mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        start, step = state["epoch"] + 1, state["step"]
    order = sample_order(train, cfg, objective.kind)
    if not order or not len(val.eligible(cfg["min_context_hours"])):
        raise ValueError("No eligible training/validation admissions")
    groups = math.ceil(len(order) / cfg["effective_batch_size"])
    warmup = math.ceil(groups * cfg["epochs"] * cfg["warmup_fraction"])
    for epoch in range(start, cfg["epochs"] + 1):
        if schedule.reductions >= cfg["stop_after_reductions"]:
            break
        torch.manual_seed(config["seed"] + epoch)
        permutation = np.random.default_rng(config["seed"] + epoch).permutation(len(order))
        model.train()
        if objective.hepa:
            objective.hepa.reset()
        total = 0.0
        for group_indices in np.array_split(permutation, groups):
            selected = [order[j] for j in group_indices]
            values = [item(train, i, draw, epoch, config["seed"], cfg, objective.kind) for i, draw in selected]
            values.sort(key=lambda x: len(x["history"]))
            micro = list(batches(values, cfg))
            optimizer.zero_grad(set_to_none=True)
            if objective.hepa:
                total += hepa_step(model, micro, objective, device)
            else:
                denominator = sum(objective.denominator(batch) for batch in micro).to(device)
                for cpu in micro:
                    n, _ = objective.statistics(model, move(cpu, device))
                    loss = objective.reduce(n, denominator)
                    loss.backward()
                    total += float(loss.detach())
            if step < warmup:
                for group in optimizer.param_groups:
                    group["lr"] = cfg["lr"] * (step+1) / warmup
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
        score, improved, stop = None, False, False
        if epoch % cfg["val_interval"] == 0:
            score = validate(model, val, cfg, objective, config["seed"], device)
            if epoch >= cfg["selection_start_epoch"]:
                improved, stop = schedule.step(score)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), schedule=schedule.state_dict(),
                     config=config, dimensions=train.dimensions, schema_key=train.schema_key, source=digest,
                     data=[train.fingerprint, val.fingerprint], stage=stage, epoch=epoch, step=step)
        save_checkpoint(out / "last.pt", **state)
        if improved:
            save_checkpoint(out / "best.pt", **state)
        log(out / "history.jsonl", phase=stage, epoch=epoch, loss=total/groups, validation_loss=score,
            lr=optimizer.param_groups[0]["lr"], reductions=schedule.reductions)
        if stop:
            break
    if not (out / "best.pt").exists():
        raise RuntimeError("No validated checkpoint selected")
    return out / "best.pt"
