# Reproduction

```bash
uv sync --extra dev
```

## Data

Place `company_years.parquet` in `data/raw/` (see the README), then build the
labels, describe the splits and summarize the data:

```bash
uv run python scripts/build_labels.py
uv run python scripts/build_temporal_splits.py
uv run python scripts/summarize_data.py
```

`build_temporal_splits.py` writes `data/temporal_splits/split_definition.json`,
`cells.txt` and `tasks_h{h}.parquet` (task membership by `emis_id` and year);
`summarize_data.py` writes row and positive counts by horizon, country and year
to `results/generated/data_summary/`.

## Baselines

```bash
uv sync --extra dev --extra neural
uv run python scripts/run_baselines.py
```

Runs every model in `configs/baselines/standard.yaml` (logistic regression,
XGBoost, LightGBM, CatBoost, RealMLP, TabM) on all horizons and test years.
Useful options: `--models xgboost catboost`, `--horizons 1 3`,
`--test-year 2014`, `--track main` for the 15 main-track tasks only,
`--n-trials 5` for a quick run, `--save-models`.

On macOS, run `realmlp`/`tabm` in a separate invocation from
`xgboost`/`lightgbm`: torch and Homebrew's `libomp` cannot share a process.

## TabPFN

```bash
uv sync --extra tabpfn
uv run --extra tabpfn python scripts/run_tabpfn.py \
  --config configs/tabpfn/vanilla_prototype_undersample.yaml

uv run --extra tabpfn python scripts/finetune_tabpfn.py \
  --config configs/tabpfn/finetune_prototype_undersample.yaml \
  --horizon 1 --test-year 2014 \
  --model-path /path/to/tabpfn.ckpt --device cuda
```

`configs/tabpfn/local_smoke.yaml` limits sample sizes for a local check.
Fine-tuning runs one split per call; loop over `data/temporal_splits/cells.txt`
to cover the grid. Each call writes per-epoch metrics, `best_epoch.json` and the
best epoch's `test_scores.parquet` to `results/generated/tabpfn_finetune/h{h}/Y{Y}/`.

## Outputs and evaluation

`run_baselines.py` writes one pair of files per model and split under
`results/generated/baselines/h{h}/Y{Y}/`: `{model}.parquet` with per-row test
scores (model, horizon, test year, `emis_id`, label, score, thresholded
prediction, unseen-firm flag) and `{model}.metrics.json` with the split's
metrics, selected hyperparameters, timings and hardware. Existing cells are
skipped unless `--overwrite` is given, so jobs can run in parallel and be
resubmitted.

Summarize with confidence intervals:

```bash
uv run python scripts/evaluate.py
uv run python scripts/evaluate.py --track full
```

This writes to `results/generated/summary/`: `{main,full}_summary.csv` (metrics
with 95% intervals per model and horizon, for all firms, seen and unseen firms
and each country), `{main,full}_paired.csv` (paired model differences) and
`per_split.csv` (one row per model and task). All models must cover the same
tasks.
