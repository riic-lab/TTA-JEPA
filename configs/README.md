# Configuration inventory
Choose a model with `--model`; use `--config` to supply an alternative YAML for that model.

| Configuration     | Purpose                                            |
| ----------------- | -------------------------------------------------- |
| `tte_jepa.yaml`   | TTE-JEPA architecture and shared training settings |
| `icarefm.yaml`    | ICareFM next-state and TTE training                |
| `hepa.yaml`       | HEPA alignment and SIGReg training                 |
| `ehrmamba.yaml`   | EHRMamba TTE training                              |
| `xgboost.yaml`    | MEDS-TAB feature extraction and XGBoost fitting    |
| `components.yaml` | Standalone defaults for component smoke checks     |
Baseline configurations inherit `tte_jepa.yaml` through `extends`. Model definitions are replaced; other configuration sections are merged by key.
Select the data profile with `--dataset joint`, `mono`, or `debug10`. The corresponding file in `datasets/` is applied after the model configuration. `debug10` runs one epoch per neural stage on a separately prepared small dataset.
Supply dataset paths, output directory, device, and the required seed through the CLI. Model configurations contain no default seed or dataset paths. See [the usage examples](../README.md#training-and-evaluation) and [the input contract](../DATA_FORMAT.md).
