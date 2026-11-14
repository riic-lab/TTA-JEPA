# Prepared-grid input contract
The CLI consumes already-normalized hourly grids, not raw EHR tables. Joint, debug10, and mono profiles use the same format with different supplied directories. Existing compatible caches can be used directly; no dependency on another source repository is required. Clinical current-state derivation and fitting preprocessing statistics belong upstream and are not reimplemented here.
## Schema directory
- `feature_schema.json`: ordered mapping of feature definitions, including `physical_columns`, `reconstruction_type`, and continuous `presence_mask_column` where applicable.
- `tte_targets.json`: ordered `targets` list of dynamic column names.
## Cache directory
Each of `train/`, `val/`, and `test/` contains `grid.safetensors`, `admissions.parquet`, and `manifest.json`. The manifest must declare `complete: true`, its `source.split`, and SHA-256 hashes `source.feature_schema_sha256` and `source.tte_targets_sha256` matching the supplied schema files. Manifests identify data during resume; the reader checks schema hashes but does not rescan entire large files to recompute their content hashes.

| Tensor     | Shape | Dtype   | Meaning                                                     |
|------------|-------|---------|-------------------------------------------------------------|
| static     | [N,S] | float32 | One static vector per admission                             |
| x_dyn      | [R,C] | float32 | Concatenated hourly model inputs                             |
| m_dyn      | [R,M] | bool    | Continuous observation then categorical presence indicators  |
| z_targets  | [R,K] | float32 | Standardized targets, NaN when not genuinely observed        |
| eep_states | [R,7] | int8    | Current clinical states: -1 unknown, 0 negative, 1 positive   |
| starts     | [N]   | int64   | Admission start offsets in hourly arrays                     |
| lengths    | [N]   | int64   | Positive hourly lengths                                     |
`admissions.parquet` provides unique `admissionid` values in the same order as `starts`; identifiers must be consistent and non-overlapping across splits. EEP state column order is circulatory failure, respiratory failure, kidney failure, liver failure, hyperglycemia, sepsis, decompensation. The final decompensation state identifies death versus live discharge. The reader maps these source columns to heads by task name, not YAML order.
## Transformations
Per-admission reads slice hourly arrays to `[T,C]`, `[T,M]`, `[T,K]`, and `[T,7]`. EEP state -1 becomes float32 NaN for label calculation; model input values are not reimputed or renormalized. Pretraining builds histories through the largest sampled anchor and pads batches to `[B,T,C]`; anchors and future windows are padded to `[B,A]` and `[B,A,48,C]`. Labels are float32 and validity is boolean. Invalid label cells contain finite zero placeholders.
No source rows are silently removed. Admissions too short for the configured minimum context plus one future hour are excluded from that phase and counted in stdout. Missing clinical outcomes are masked, not converted to negative labels. The debug10 profile expects its own prepared dataset and does not select ten admissions from another input implicitly.
