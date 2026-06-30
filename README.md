# V4FinBench

V4FinBench is a benchmark for corporate financial distress prediction in Visegrad Group economies. It accompanies the paper **"V4FinBench: Benchmarking Tabular Foundation Models, LLMs, and Standard Methods on Corporate Bankruptcy Prediction"**.

The repository is intended to reproduce the public benchmark pipeline and reference evaluations from the released Kaggle data. It does **not** reproduce the original EMIS extraction process because the raw EMIS files are not redistributed.

## License

The code in this repository is released under the MIT License. The released
V4FinBench dataset hosted on Kaggle is licensed under **CC BY 4.0**.

## Public Data

The canonical public dataset is hosted on Kaggle:

https://www.kaggle.com/datasets/sebastiantomczak10/v4-group-corporate-bankruptcy/data

Dataset license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Expected files:

| File | Meaning |
| --- | --- |
| `company_years.parquet` | Unlabeled company-year records |
| `company_years_h1.parquet` | Horizon file 1, paper horizon `h=0`, current-year distress |
| `company_years_h2.parquet` | Horizon file 2, paper horizon `h=1`, one-year-ahead distress |
| `company_years_h3.parquet` | Horizon file 3, paper horizon `h=2`, two-year-ahead distress |
| `company_years_h4.parquet` | Horizon file 4, paper horizon `h=3`, three-year-ahead distress |
| `company_years_h5.parquet` | Horizon file 5, paper horizon `h=4`, four-year-ahead distress |
| `company_years_h6.parquet` | Horizon file 6, paper horizon `h=5`, five-year-ahead distress |

You can either download the files manually from Kaggle and place them under `data/raw/`, or use the download script once it is added:

```bash
uv run python scripts/download_kaggle.py --out data/raw
```

Manual layout:

```text
data/raw/
├── company_years.parquet
├── company_years_h1.parquet
├── company_years_h2.parquet
├── company_years_h3.parquet
├── company_years_h4.parquet
├── company_years_h5.parquet
└── company_years_h6.parquet
```

## What Is Reproducible

The repository should make the following reproducible from public artifacts:

1. loading the Kaggle parquet files,
2. verifying or regenerating horizon labels from `company_years.parquet`,
3. generating deterministic train/validation/test folds,
4. running the standard tabular baselines,
5. running TabPFN fine-tuning and evaluation,
6. running the separate Llama-3-8B QLoRA experiment,
7. aggregating results into paper-style tables.

The original EMIS reconstruction is documented as data provenance only. The raw EMIS source files are not shared, so scripts for raw Excel ingestion should not be part of the required reproduction path.

## Evaluation Protocol

The benchmark uses five grouped folds. All observations from a company must stay in the same fold, and fold construction must preserve the country structure.

Fold generation must use the fixed seed used for the paper:

```text
n_splits = 5
random_state = 42
group_col = company
country_col = country
```

For each run, one fold is used for validation, the next fold is used for testing, and the remaining three folds are used for training:

```text
val_fold = fold
test_fold = (fold + 1) % 5
train_folds = all other folds
```

The generated fold files should therefore be identical for all users given the same Kaggle parquet files and seed.

## Models Included

The standard tabular approaches should come from the `economic-data` implementation path, not from the `financial-distress-foundational-models` XGBoost-only code. The benchmark should include:

- logistic regression,
- multilayer perceptron,
- random forest,
- XGBoost,
- LightGBM,
- CatBoost.

TabPFN fine-tuning should come from the foundation-model code, but with hard-coded cluster paths removed and the data/fold paths made configurable.

The old Llama evaluation code from `economic-data` is not used. Llama-3-8B
QLoRA fine-tuning is implemented as a separate experiment under
`src/v4finbench/llama`, `configs/llama`, and `scripts/llama_*`.

## Repository Structure

```text
V4FinBench/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── data.yaml
│   ├── baselines/
│   ├── llama/
│   └── tabpfn/
├── data/
│   ├── raw/
│   ├── processed/
│   └── folds/
├── docs/
│   ├── data_provenance.md
│   ├── llama_experiment.md
│   ├── benchmark_protocol.md
│   └── reproduction.md
├── scripts/
│   ├── aggregate_finetune_best.py
│   ├── aggregate_results.py
│   ├── download_kaggle.py
│   ├── build_labels.py
│   ├── build_folds.py
│   ├── finetune_tabpfn.py
│   ├── llama_eval.py
│   ├── llama_prepare_data.py
│   ├── llama_threshold.py
│   ├── llama_train_qlora.py
│   ├── reproduce_*.sh
│   ├── run_baselines.py
│   ├── run_tabpfn.py
│   ├── summarize_data.py
│   └── verify_labels.py
├── src/
│   └── v4finbench/
│       ├── data/
│       │   ├── io.py
│       │   ├── labels.py
│       │   ├── folds.py
│       │   └── preprocessing.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── thresholds.py
│       │   └── protocol.py
│       ├── llama/
│       │   ├── formatting.py
│       │   ├── inference.py
│       │   ├── metrics.py
│       │   └── sampling.py
│       ├── models/
│       │   ├── baselines.py
│       │   ├── tabpfn.py
│       │   └── tabpfn_finetune.py
│       └── sampling/
│           ├── strategies.py
│           └── prototypes.py
├── tests/
├── slurm/
└── results/
```

## Development

This project should use `uv`.

```bash
uv sync --extra dev
uv run --extra dev pytest
```

Suggested reproduction commands once the scripts are in place:

```bash
uv run python scripts/build_labels.py --input data/raw/company_years.parquet --out data/processed
uv run python scripts/verify_labels.py --generated data/processed --reference data/raw
uv run python scripts/summarize_data.py --data-dir data/raw --out results/generated/data_summary.csv
uv run python scripts/build_folds.py --data-dir data/raw --out data/folds --seed 42
uv run python scripts/run_baselines.py --data-dir data/raw --folds-dir data/folds
uv run python scripts/run_tabpfn.py --data-dir data/raw --folds-dir data/folds
uv run python scripts/aggregate_results.py --input results/generated/baselines/metrics.csv --out results/generated/baselines/summary.csv
```

For a quick baseline smoke run before launching the full grid:

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

Aggregate baseline metrics after running multiple folds:

```bash
uv run python scripts/aggregate_results.py \
  --input results/generated/baselines/metrics.csv \
  --out results/generated/baselines/summary.csv
```

Run a local vanilla TabPFN smoke test after installing the optional TabPFN
dependencies. Keep the context small for local checks.

```bash
uv sync --extra tabpfn --extra rapids
uv run --extra tabpfn --extra rapids python scripts/run_tabpfn.py \
  --config configs/tabpfn/local_smoke.yaml \
  --data-dir data/raw \
  --folds-dir data/folds \
  --horizon 0 \
  --fold 0
```

To evaluate a specific TabPFN checkpoint or weights file, add:

```bash
--model-path /path/to/tabpfn_checkpoint.ckpt
```

Fine-tune TabPFN for one horizon/fold:

```bash
uv run --extra tabpfn --extra rapids python scripts/finetune_tabpfn.py \
  --config configs/tabpfn/finetune_prototype_undersample.yaml \
  --data-dir data/raw \
  --folds-dir data/folds \
  --horizon 0 \
  --fold 0 \
  --model-path /path/to/tabpfn_checkpoint.ckpt \
  --device cuda
```

Fine-tune one horizon-conditioned TabPFN model for a fold by joining all
horizon splits and adding `prediction_horizon` as a numeric feature:

```bash
uv run --extra tabpfn --extra rapids python scripts/finetune_tabpfn_horizon_conditioned.py \
  --config configs/tabpfn/finetune_horizon_conditioned.yaml \
  --data-dir data/raw \
  --folds-dir data/folds \
  --horizons all \
  --fold 0 \
  --model-path /path/to/tabpfn_checkpoint.ckpt \
  --device cuda
```

Prototype undersampling uses `prototype_backend`; the prototype TabPFN configs
default to `cuml`, which requires the `rapids` extra and runs both KMeans and
nearest-row lookup on GPU. Use `--prototype-backend sklearn` to force CPU.

Aggregate fine-tuning best epochs:

```bash
uv run python scripts/aggregate_finetune_best.py \
  --root results/generated/tabpfn_finetune \
  --out results/generated/tabpfn_finetune/best_epochs.csv \
  --summary results/generated/tabpfn_finetune/summary.csv
```

Prepare the separate Llama QLoRA experiment data:

```bash
uv run python scripts/llama_prepare_data.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  --data-dir data/raw \
  --out data/llama \
  --horizon 0
```

The Llama system prompt is configurable in `configs/llama/qlora_llama3_8b.yaml`
or via `--system-prompt` / `--system-prompt-file`. The default asks whether a
company will go bankrupt within `{horizon_years}` year(s), not whether it will
file a legal bankruptcy case.

Train and evaluate a Llama adapter. This path requires GPU infrastructure and
the optional Llama dependencies. Train one separate adapter per horizon dataset;
do not train one shared adapter across all six horizons.

```bash
uv sync --extra llama
uv run --extra llama python scripts/llama_train_qlora.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  --train-file data/llama/llama_h0_train.csv \
  --output-dir results/generated/llama/h0_adapter

uv run --extra llama python scripts/llama_eval.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --adapter-path results/generated/llama/h0_adapter \
  --test-file data/llama/llama_h0_test.csv \
  --out results/generated/llama/h0_predictions.csv \
  --compute-yes-no-probs
```
