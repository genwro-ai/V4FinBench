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

`build_labels.py` takes `--mode within` for the alternative label definition and
`--keep-censored` to keep censored rows as negatives. `build_temporal_splits.py`
writes `data/temporal_splits/split_definition.json` and `cells.txt`;
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
`--test-year 2014`, `--n-trials 5` for a quick run, `--save-models`.

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

Every runner writes per-split metrics to `metrics.csv` and per-row test scores
to parquet files (model, horizon, test year, `emis_id`, label, score, thresholded
prediction). Summarize them with confidence intervals:

```bash
uv run python scripts/evaluate.py --scores results/generated
uv run python scripts/evaluate.py --scores results/generated --all-years
```

This writes `{primary,all_years}_summary.csv` (metrics with 95% intervals per
model and horizon) and `{primary,all_years}_paired.csv` (paired model
differences) to `results/generated/summary/`.
