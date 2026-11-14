import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import polars as pl
import torch
import yaml
from safetensors.torch import save_file
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", choices=("tte_jepa", "icarefm", "hepa", "ehrmamba", "xgboost"), default="tte_jepa")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / f"{args.model}.yaml")
    if args.model == "tte_jepa":
        config["model"]["student"].update(width=16, n_layers=1, n_heads=4, n_kv_heads=4)
        config["model"]["teacher"].update(width=8, n_heads=2)
        config["model"]["tte_head"]["hidden_dim"] = 16
    elif args.model != "xgboost":
        config["model"].update(width=64 if args.model == "ehrmamba" else 16, n_layers=1)
        for key in ("n_heads", "n_kv_heads"):
            if key in config["model"]:
                config["model"][key] = 4
        if "tte_hidden" in config["model"]:
            config["model"]["tte_hidden"] = 16
    if args.model == "xgboost":
        config["xgboost"].update(num_boost_round=3, early_stopping_rounds=2, min_child_weight=0)
    config["pretraining"].update(batch_size=2, anchors_per_admission=2)
    if "tte_training" in config:
        config["tte_training"]["anchors_per_admission"] = 2
    config["downstream"]["hidden_dim"] = 16
    config["finetuning"].update(batch_rows=4, predict_batch_rows=4, anchor_chunk_size=4)
    config["cpu_threads"] = 1
    with tempfile.TemporaryDirectory(prefix="tte-jepa-smoke-") as temporary:
        path = Path(temporary)
        schema = {"age": {"physical_columns": ["age"], "reconstruction_type": "static_numeric"},
                  "value": {"physical_columns": ["value"], "reconstruction_type": "direct_numeric", "presence_mask_column": "observed"}}
        (path / "feature_schema.json").write_text(json.dumps(schema))
        (path / "tte_targets.json").write_text(json.dumps({"targets": ["value"]}))
        hashes = {f"{name}_sha256": hashlib.sha256((path / f"{name}.json").read_bytes()).hexdigest()
                  for name in ("feature_schema", "tte_targets")}
        for split in ("train", "val", "test"):
            directory = path / "cache" / split
            directory.mkdir(parents=True)
            events = torch.zeros(128, 7, dtype=torch.int8)
            events[63, :] = 1
            data = {"starts": torch.tensor([0, 64]), "lengths": torch.tensor([64, 64]),
                    "static": torch.zeros(2, 1), "x_dyn": torch.arange(128).float()[:, None] / 64,
                    "m_dyn": torch.ones(128, 1, dtype=torch.bool),
                    "z_targets": torch.arange(128).float()[:, None] / 64, "eep_states": events}
            save_file(data, directory / "grid.safetensors")
            pl.DataFrame({"admissionid": [f"{split}-a", f"{split}-b"]}).write_parquet(directory / "admissions.parquet")
            (directory / "manifest.json").write_text(json.dumps({"complete": True, "source": {"split": split, **hashes}}))
        (path / "config.yaml").write_text(yaml.safe_dump(config))
        command = [sys.executable, "-m", "CLI.train", "run", "--dataset", "debug10", "--data-dir", str(path),
                   "--cache-dir", str(path / "cache"), "--output-dir", str(path / "run"), "--config", str(path / "config.yaml")]
        command += ["--seed", str(args.seed)]
        command += ["--model", args.model, "--device", args.device]
        subprocess.run(command, cwd=root, check=True)
        result = pl.read_parquet(path / "run/evaluation/test_predictions.parquet")
        assert result.height == 120
        assert result["split"].unique().to_list() == ["test"]
        assert result["decompensation/valid"].all()
        assert result["decompensation/label"].sum() == 24
        subprocess.run(command + ["--resume"], cwd=root, check=True)
        print("PASS: prepared data → pretraining → frozen-head training → test export → resume")


if __name__ == "__main__":
    main()
