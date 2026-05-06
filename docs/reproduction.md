# Reproduction

Install dependencies with `uv`:

```bash
uv sync --extra dev
```

Download the Kaggle files manually into `data/raw/`, or use:

```bash
uv sync --extra kaggle
uv run python scripts/download_kaggle.py --out data/raw
```

Generate deterministic folds:

```bash
uv run python scripts/build_labels.py --input data/raw/company_years.parquet --out data/processed
uv run python scripts/verify_labels.py --generated data/processed --reference data/raw
uv run python scripts/summarize_data.py --data-dir data/raw --out results/generated/data_summary.csv
uv run python scripts/build_folds.py --data-dir data/raw --out data/folds --seed 42
```

The first label command regenerates the six horizon files into `data/processed/`.
The second compares the existing regenerated files against the released Kaggle
files in `data/raw/` without recalculating labels.

Each horizon fold directory includes:

```text
fold_assignments.txt
split_0_train_idx.txt
split_0_val_idx.txt
split_0_test_idx.txt
...
metadata.json
```

`metadata.json` records the seed, fold sizes, full-dataset class counts, and
train/validation/test class counts for each fold rotation.

All model runners should use the shared preprocessing and evaluation utilities
under `src/v4finbench/`: training-only median imputation, training-only
standardization, validation F1 threshold calibration, and test-set metric
reporting.

Run a small baseline smoke test:

```bash
uv run python scripts/run_baselines.py \
  --data-dir data/raw \
  --folds-dir data/folds \
  --horizon 0 \
  --fold 0 \
  --model logistic_regression \
  --max-candidates 1 \
  --no-save-model
```

`run_baselines.py` replaces `metrics.csv` and `metrics.json` by default so
reruns do not silently duplicate rows. Use `--append` to add more runs to an
existing metrics file.

Aggregate per-fold baseline metrics:

```bash
uv run python scripts/aggregate_results.py \
  --input results/generated/baselines/metrics.csv \
  --out results/generated/baselines/summary.csv
```

Run a local vanilla TabPFN smoke test. Keep the context and split samples small
locally; full paper-scale TabPFN runs are much heavier.

```bash
uv sync --extra tabpfn
uv run --extra tabpfn python scripts/run_tabpfn.py \
  --config configs/tabpfn/local_smoke.yaml \
  --data-dir data/raw \
  --folds-dir data/folds \
  --horizon 0 \
  --fold 0
```

For paper-scale vanilla TabPFN context evaluation, remove the max-sample smoke
limits and use `--n-context-samples 10000`.

To evaluate a specific TabPFN checkpoint or weights file, pass `--model-path`
or set `model_path` in the YAML config. The value is forwarded to
`TabPFNClassifier(model_path=...)`.

Fine-tune TabPFN for one horizon/fold:

```bash
uv run --extra tabpfn python scripts/finetune_tabpfn.py \
  --config configs/tabpfn/finetune_prototype_undersample.yaml \
  --data-dir data/raw \
  --folds-dir data/folds \
  --horizon 0 \
  --fold 0 \
  --model-path /path/to/tabpfn_checkpoint.ckpt \
  --device cuda
```

The fine-tuning script writes per-epoch `metrics.csv`, `metrics.json`, and
`best_epoch.json` under `results/generated/tabpfn_finetune/h{horizon}/fold_{fold}`.

Aggregate best fine-tuning epochs across folds and horizons:

```bash
uv run python scripts/aggregate_finetune_best.py \
  --root results/generated/tabpfn_finetune \
  --out results/generated/tabpfn_finetune/best_epochs.csv \
  --summary results/generated/tabpfn_finetune/summary.csv
```
