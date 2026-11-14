import argparse
from pathlib import Path

import torch
import yaml

from training.pipeline import run
from utils.config import load_config
from utils.data import Grid


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="TTE-JEPA training on prepared hourly grids")
    parser.add_argument("stage", choices=("pretrain", "tte", "finetune", "evaluate", "run"))
    parser.add_argument("--model", choices=("tte_jepa", "icarefm", "hepa", "ehrmamba", "xgboost"), default="tte_jepa")
    parser.add_argument("--dataset", choices=("joint", "debug10", "mono"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Feature schema and TTE target inventory")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Prepared train/val/test grid caches")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Pretrained checkpoint for finetune; downstream checkpoint for evaluate")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.model == "ehrmamba" and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
        parser.error("EHRMamba requires a CUDA device for the Mamba3 kernels")
    config = load_config(args.config or root / "configs" / f"{args.model}.yaml")
    if config.get("method", "tte_jepa") != args.model:
        parser.error("--model and configuration method differ")
    profile = yaml.safe_load((root / "configs/datasets" / f"{args.dataset}.yaml").read_text())
    for section, settings in profile.items():
        if isinstance(settings, dict):
            config[section].update(settings)
        else:
            config[section] = settings
    if "tte_training" in config and args.dataset == "debug10":
        config["tte_training"].update(epochs=1, val_interval=1, selection_start_epoch=1)
    horizon = config["model"].get("horizon", config["model"].get("teacher", {}).get("horizon", 48))
    if horizon != 48:
        raise ValueError("Prepared-grid label protocol requires a 48-hour horizon")
    config["seed"] = args.seed
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = args.output_dir / "config.yaml"
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != config:
        raise ValueError("Output directory already has a different configuration")
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    def grid(split):
        return Grid(args.data_dir, args.cache_dir, split)
    run(args, config, grid)


if __name__ == "__main__":
    main()
