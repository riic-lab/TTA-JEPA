import math

import numpy as np
import torch

from loss.reduction import admission_statistics
from loss.tte_jepa import TTEJEPALoss
from model.tte_jepa import TTEJEPA
from utils.sampling import collate, sample
from .runtime import Plateau, log, move, save_checkpoint, optimizer_for


def microbatches(items, config):
    pending, longest = [], 0
    for item in items:
        size = len(item["history"]) + 48 * len(item["anchors"])
        if pending and (len(pending) >= config["batch_size"] or
                        max(longest, size) * (len(pending) + 1) > config["max_batch_tokens"]):
            yield collate(pending)
            pending, longest = [], 0
        pending.append(item)
        longest = max(longest, size)
    if pending:
        yield collate(pending)


def denominators(batches, cap):
    jepa, tte = 0, 0
    for batch in batches:
        valid = batch["anchor_valid"][..., None].expand(-1, -1, 48)
        _, jd = admission_statistics(torch.zeros_like(valid, dtype=torch.float32), valid,
                                     batch["lengths"], cap, (1, 2))
        valid = batch["tte_valid"]
        _, td = admission_statistics(torch.zeros_like(valid, dtype=torch.float32), valid,
                                     batch["lengths"], cap, (1, 3))
        jepa, tte = jepa + jd, tte + td
    return jepa, tte


@torch.no_grad()
def validate(model, grid, indices, config, objective, device, seed):
    model.eval()
    total = None
    items = (sample(grid, i, config, seed, 0, draw)
             for i in indices for draw in range(config["validation_draws"]))
    for batch in microbatches(items, config):
        batch = move(batch, device)
        stats = objective.statistics(model(batch), batch)
        total = stats if total is None else {k: total[k] + v for k, v in stats.items()}
    return {k: float(v) for k, v in objective.reduce(total).items()}


def pretrain(train, val, config, out, device, resume=False):
    cfg = config["pretraining"]
    indices, val_indices = train.eligible(cfg["min_context_hours"]), val.eligible(cfg["min_context_hours"])
    if not len(indices) or not len(val_indices):
        raise ValueError("No eligible admissions")
    cap = float(np.quantile(train.lengths, 0.9))
    model = TTEJEPA(**train.dimensions, **config["model"]).to(device)
    optimizer = optimizer_for(model, cfg)
    schedule = Plateau(optimizer, cfg)
    objective = TTEJEPALoss(cap)
    out.mkdir(parents=True, exist_ok=True)
    start, step = 1, 0
    if (out / "last.pt").exists():
        if not resume:
            raise FileExistsError("Pretraining exists; use --resume or a new output directory")
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if state["config"] != config or state["data"] != [train.fingerprint, val.fingerprint]:
            raise ValueError("Resume configuration or data mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        start, step = state["epoch"] + 1, state["step"]
    groups = math.ceil(len(indices) / cfg["effective_batch_size"])
    warmup = math.ceil(groups * min(cfg["epochs"] * cfg["warmup_fraction"], 5))
    for epoch in range(start, cfg["epochs"] + 1):
        if schedule.reductions >= cfg["stop_after_reductions"]:
            break
        torch.manual_seed(config["seed"] + epoch)
        order = np.random.default_rng(config["seed"] + epoch).permutation(indices)
        model.train()
        total_loss = 0.0
        for offset in range(0, len(order), cfg["effective_batch_size"]):
            group = order[offset:offset + cfg["effective_batch_size"]]
            items = [sample(train, i, cfg, config["seed"], epoch) for i in group]
            batches = list(microbatches(items, cfg))
            jd, td = [x.to(device) for x in denominators(batches, cap)]
            optimizer.zero_grad(set_to_none=True)
            for cpu in batches:
                batch = move(cpu, device)
                stats = objective.statistics(model(batch), batch)
                tte = (stats["tte_num"] / td.clamp_min(1e-12) * (td > 0)).sum() / (td > 0).sum().clamp_min(1)
                loss = stats["jepa_num"] / jd.clamp_min(1) + tte
                loss.backward()
                total_loss += float(loss.detach())
            if step < warmup:
                for param in optimizer.param_groups:
                    param["lr"] = cfg["lr"] * (step + 1) / warmup
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            model.update_target()
            step += 1
        improved, stop, scores = False, False, {}
        if epoch % cfg["val_interval"] == 0:
            scores = validate(model, val, val_indices, cfg, objective, device, config["seed"])
            if epoch >= cfg["selection_start_epoch"]:
                improved, stop = schedule.step(scores["loss"])
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), schedule=schedule.state_dict(),
                     epoch=epoch, step=step, config=config, dimensions=train.dimensions, los_cap=cap,
                     data=[train.fingerprint, val.fingerprint], schema_key=train.schema_key)
        save_checkpoint(out / "last.pt", **state)
        if improved:
            save_checkpoint(out / "best.pt", **state)
        log(out / "history.jsonl", phase="pretraining", epoch=epoch, loss=total_loss/groups,
            validation=scores, lr=optimizer.param_groups[0]["lr"], reductions=schedule.reductions)
        if stop:
            break
    if not (out / "best.pt").exists():
        raise RuntimeError("No validated checkpoint selected; check epoch/validation settings")
    return out / "best.pt"
