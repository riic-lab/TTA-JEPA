import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from safetensors import safe_open


class Grid:
    """Lazy reader for prepared hourly grids; clinical preprocessing stays upstream."""

    def __init__(self, data_dir, cache_dir, split):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        self.split = split
        self.directory = Path(cache_dir) / split
        self.schema = json.loads((Path(data_dir) / "feature_schema.json").read_text())
        self.targets = json.loads((Path(data_dir) / "tte_targets.json").read_text())["targets"]
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if not manifest.get("complete") or manifest["source"]["split"] != split:
            raise ValueError("Incomplete cache or split mismatch")
        for filename in ("feature_schema", "tte_targets"):
            digest = hashlib.sha256((Path(data_dir) / f"{filename}.json").read_bytes()).hexdigest()
            if digest != manifest["source"][f"{filename}_sha256"]:
                raise ValueError("Cache and input schema differ")
        self.schema_key = [manifest["source"][f"{name}_sha256"] for name in ("feature_schema", "tte_targets")]
        self.fingerprint = hashlib.sha256((self.directory / "manifest.json").read_bytes()).hexdigest()
        self.handle = safe_open(self.directory / "grid.safetensors", framework="pt", device="cpu")
        self.starts = self.handle.get_tensor("starts").numpy()
        self.lengths = self.handle.get_tensor("lengths").numpy()
        identities = pl.read_parquet(self.directory / "admissions.parquet")
        self.ids = identities["admissionid"].to_list()
        if len(self.ids) != len(self.lengths) or len(set(self.ids)) != len(self.ids):
            raise ValueError("Admission identities are not unique or aligned")
        dynamic, continuous = [], []
        for spec in self.schema.values():
            if spec["reconstruction_type"].startswith("static_"):
                continue
            columns = spec["physical_columns"]
            if "presence_mask_column" in spec:
                if len(columns) != 1:
                    raise ValueError("A continuous observation mask must identify one channel")
                continuous.append(len(dynamic))
            dynamic.extend(columns)
        self.continuous = torch.tensor(continuous, dtype=torch.long)
        self.target_indices = torch.tensor([dynamic.index(t) for t in self.targets])
        shapes = {k: self.handle.get_slice(k).get_shape() for k in ("static", "x_dyn", "m_dyn", "z_targets", "eep_states")}
        if len(self.lengths) == 0 or (self.lengths <= 0).any():
            raise ValueError("Prepared split must contain positive admission lengths")
        expected_starts = np.r_[0, np.cumsum(self.lengths[:-1])]
        if not np.array_equal(self.starts, expected_starts):
            raise ValueError("Admission offsets must describe contiguous hourly rows")
        if shapes["static"][0] != len(self) or any(shapes[k][0] != int(self.lengths.sum())
                                                  for k in shapes if k != "static"):
            raise ValueError("Prepared tensor row counts are not aligned")
        if shapes["x_dyn"][1] != len(dynamic):
            raise ValueError("Dynamic schema width differs from cache")
        self.dimensions = dict(n_static=shapes["static"][1], n_values=shapes["x_dyn"][1],
                               n_indicators=shapes["m_dyn"][1], n_targets=len(self.targets))
        if shapes["z_targets"][1] != len(self.targets) or shapes["eep_states"][1] != 7:
            raise ValueError("Unexpected target layout")
        print(json.dumps({"split": split, "admissions": len(self.ids), "hours": int(self.lengths.sum()),
                          "dimensions": self.dimensions, "records_dropped": 0}), flush=True)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        start, length = int(self.starts[index]), int(self.lengths[index])
        item = {name: self.handle.get_slice(key)[start:start + length]
                for name, key in (("values", "x_dyn"), ("indicators", "m_dyn"),
                                  ("trace", "z_targets"), ("events", "eep_states"))}
        item["static"] = self.handle.get_slice("static")[index]
        item["events"] = item["events"].float()
        item["events"][item["events"] == -1] = float("nan")
        return item

    def eligible(self, min_context):
        indices = np.flatnonzero(self.lengths >= min_context + 1)
        print(f"{self.split}: {len(self) - len(indices)} admissions excluded for insufficient context", flush=True)
        return indices
