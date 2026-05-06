# Llama-3-8B QLoRA Experiment

The Llama experiment is separate from the standard tabular benchmark and TabPFN
paths. It lives under `src/v4finbench/llama`, `configs/llama`, and
`scripts/llama_*`.

The public implementation follows the QLoRA setup from the paper:

- `meta-llama/Meta-Llama-3-8B` base model,
- 20,000 sampled training observations per horizon,
- 2% validation split inside `scripts/llama_train_qlora.py`,
- up to 100,000 sampled test observations per horizon,
- a 10% positive-class sampling target where enough positives are available,
- fixed seed 42 for data sampling and training,
- structured company-year records serialized into chat columns,
- `YES`/`NO` assistant labels,
- optional token-probability evaluation over `YES` and `NO`.

Each horizon dataset is treated as its own experiment. Prepare all horizon CSVs
if needed, but train one adapter per horizon, for example `h0_adapter`,
`h1_adapter`, ..., `h5_adapter`.

The public horizons are highly imbalanced. Data preparation samples without
replacement, builds the training split first, and then fills the test split from
the remaining rows. If a requested positive-class ratio cannot be satisfied, the
script uses all available positives for that split and fills the rest with
negative examples up to the requested split size.

The legacy `economic-data/scripts/llama_evaluation.py` path is intentionally not
used. It relied on an older evaluation regime and mixed concerns that are now
split into preparation, QLoRA training, adapter evaluation, and thresholding.

## Data Preparation

Prepare one horizon:

```bash
uv run python scripts/llama_prepare_data.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  --data-dir data/raw \
  --out data/llama \
  --horizon 0
```

Prepare all horizons:

```bash
uv run python scripts/llama_prepare_data.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  --data-dir data/raw \
  --out data/llama
```

Outputs:

```text
data/llama/llama_h0_train.csv
data/llama/llama_h0_test.csv
...
data/llama/llama_h5_train.csv
data/llama/llama_h5_test.csv
```

Each row contains `year`, `country`, `emis_id`, `system`, `user`,
`main_label`, and `assistant`.

The system prompt is parameterized through `configs/llama/qlora_llama3_8b.yaml`,
`--system-prompt`, or `--system-prompt-file`. The default public prompt follows
the paper horizon convention: `h=0` means the current reporting year, while
`h=1` through `h=5` mean one through five years after the observed reporting
year. Prompt templates support `{horizon}`, `{horizon_years}`, and
`{horizon_description}` placeholders.

## Training

Install optional dependencies and run QLoRA:

```bash
uv sync --extra llama
uv run --extra llama python scripts/llama_train_qlora.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  --train-file data/llama/llama_h0_train.csv \
  --output-dir results/generated/llama/h0_adapter
```

Repeat the command with the corresponding `llama_h{i}_train.csv` and
`h{i}_adapter` output directory for each horizon. The original experiment does
not train one shared Llama adapter across all horizon datasets.

On SLURM:

```bash
sbatch slurm/llama_train_qlora.sbatch \
  --train-file data/llama/llama_h0_train.csv \
  --output-dir results/generated/llama/h0_adapter
```

## Evaluation

```bash
uv run --extra llama python scripts/llama_eval.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --adapter-path results/generated/llama/h0_adapter \
  --test-file data/llama/llama_h0_test.csv \
  --out results/generated/llama/h0_predictions.csv \
  --compute-yes-no-probs
```

Threshold the `p_yes` scores by validation/test F1 when probability columns are
available:

```bash
uv run python scripts/llama_threshold.py \
  --input results/generated/llama/h0_predictions.csv \
  --out results/generated/llama/h0_thresholded.csv
```
