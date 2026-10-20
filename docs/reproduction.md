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

## Foundation models

TabPFN-3.5 (`tabpfn==9.2.0`) and TabICL v2 (`tabicl==2.2.0`) need a GPU. The
TabPFN-3.5 weights require an accepted license and a Prior Labs API key
(`TABPFN_TOKEN`) on first download; on offline nodes, copy the weight cache and
set `TABPFN_MODEL_CACHE_DIR`.

In-context (no training) runs through the baseline runner with its own config:
TabPFN and TabICL, each with the full training set, random undersampling,
random oversampling or prototype undersampling as context.

```bash
uv sync --extra tabpfn
uv run python scripts/run_baselines.py \
  --config configs/foundation/in_context.yaml --track main
```

Fine-tuning runs one estimator and sampling variant per call (sampling: `none`,
`undersample`, `prototype_undersampling`). For each task it fine-tunes for up to
10 epochs per learning rate in `configs/foundation/finetune.yaml`, early
stopping on the validation years, keeps the learning rate with the best
validation PR-AUC and saves that run's best-epoch weights next to its results.

```bash
uv run python scripts/finetune_foundation.py --estimator tabpfn --sampling none
```

## Athena (Slurm)

Stage the raw data and weights once under `$SCRATCH/V4FinBench-assets`
(`company_years.parquet`, `tabpfn/tabpfn-v3.5-20260909.safetensors`, and the
TabICL checkpoint in a Hugging Face cache under `hf/`). `slurm/athena_env.sh`
points the libraries at them in offline mode. In an interactive GPU allocation,
from the checkout:

```bash
bash slurm/setup.sh
```

This installs the environment, builds labels and task files (including
`cells_main.txt` for the main track) and checks that both models load. Then
submit one array per model, one element per task:

```bash
RUNNER=baseline CONFIG=configs/foundation/in_context.yaml MODEL=tabpfn_full \
  sbatch --array=0-14 slurm/task.sbatch
RUNNER=finetune ESTIMATOR=tabpfn SAMPLING=none sbatch --array=0-14 slurm/task.sbatch
```

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
