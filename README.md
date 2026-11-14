# TTE-JEPA
Time-to-event-grounded joint-embedding prediction for ICU time series.
TTE-JEPA learns a patient-history representation by predicting representations of future observations. A separate teacher is trained with time-to-event supervision; its exponential moving average provides stop-gradient targets for the history encoder. Downstream models predict seven clinical outcomes.
## Model
The default configuration uses a three-layer, 512-dimensional history Transformer with 48 horizon queries. The teacher has its own timestep encoder and a single causal, 256-dimensional Transformer block over future hourly observations. It uses neither learned query tokens nor cross-attention. Both encoders concatenate values with observation indicators.
Pretraining combines a normalized representation-matching loss with time-to-event binary cross-entropy. Downstream training freezes the history encoder and fits its projection and task-specific prefix heads. Checkpoints are selected on validation data; test data are reserved for evaluation.
Architecture details are in [ARCHITECTURE.md](ARCHITECTURE.md), with defaults in [configs/tte_jepa.yaml](configs/tte_jepa.yaml).
## Baselines
- **ICareFM:** next-state pretraining, then TTE training on the last observed state, followed by frozen-state EEP heads.
- **HEPA:** a shared history/future Transformer with L1 alignment and sliced SIGReg; independent horizon predictors for downstream tasks.
- **EHRMamba:** six SISO Mamba-3 blocks, TTE pretraining, and frozen-state EEP heads.
- **XGBoost:** seven classifiers on causal MEDS-TAB summary features, with admission weighting and validation-based early stopping.
All models use the same prepared splits, endpoint labels, and admission-balanced test metrics. TTE queries use threshold sign to determine crossing direction; conditioning contains threshold and variable identity only.
## Installation
Install the dependencies in a Python environment from the repository root:
```bash
pip install -r requirements.txt
```
The implementation requires PyTorch 2.5 or later. Dataset files and pretrained weights are not included.
EHRMamba additionally requires a CUDA-compatible `mamba-ssm` installation exposing `Mamba3`; see [requirements-mamba.txt](requirements-mamba.txt). Other models do not import Mamba kernels.
## Data and usage
Inputs are prepared, normalized hourly grids with explicit training, validation, and test splits. Raw EHR preprocessing is outside this repository. See [DATA_FORMAT.md](DATA_FORMAT.md) for the schema and cache format.
Run pretraining, downstream fitting, and test evaluation:
```bash
python -m CLI.train run \
  --dataset joint --seed "$SEED" \
  --data-dir /path/to/schema \
  --cache-dir /path/to/prepared_grids \
  --output-dir /path/to/run
```
Set `SEED` to an integer of your choice. All entry points require an explicit `--seed`; no seed value is supplied in repository configuration. The chosen value is saved only in local run metadata for resume and reproducibility.
Dataset profiles are `joint`, `mono`, and `debug10`. The debug profile expects a separately prepared small dataset. Execution defaults to CPU; select a device with `--device`.
Select a baseline with `--model icarefm`, `hepa`, `ehrmamba`, or `xgboost`. Each loads its corresponding YAML in `configs/`; `--config` selects an override. `run` executes all required stages. ICareFM adds a `tte` stage between `pretrain` and `finetune`. XGBoost uses `finetune` for classifier fitting and has no representation-pretraining stage.
Use `pretrain`, `finetune`, or `evaluate` instead of `run` for an individual stage. Supply `--checkpoint` for pretrained or downstream weights, and `--resume` to resume training. XGBoost resume retains completed task models and restarts unfinished tasks. A synthetic end-to-end check is available:
```bash
python -m CLI.smoke_pipeline --seed "$SEED"
```
The smoke CLI accepts the same `--model` selector. EHRMamba execution requires `--device cuda`; all other smokes can run on CPU.
Runs save resolved configuration, checkpoints, JSONL training logs, representation caches, and evaluation metrics with Parquet predictions.
## Repository structure
- `model/`: history encoder, future teacher, and prediction heads.
- `training/`: pretraining, downstream fitting, caching, and EMA updates.
- `evaluation/`: inference and clinical prediction metrics.
- `loss/`: objectives and admission-weighted reductions.
- `utils/`: data access, sampling, and labels.
- `configs/`: model, training, and dataset YAML settings.
- `CLI/`: command-line entry points.
- `baselines/`: baseline architectures, readouts, and tabular features.
## Implementation status
The CLI provides single-device training, checkpoint resume, and test prediction export. Downstream CUDA fitting uses BF16 autocast with FP32 losses; CPU fitting uses FP32. Distributed training and bootstrap confidence intervals are not included.
