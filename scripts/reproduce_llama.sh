#!/usr/bin/env bash
set -euo pipefail

HORIZON="${1:-0}"

uv run python scripts/llama_prepare_data.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  --data-dir data/raw \
  --out data/llama \
  --horizon "${HORIZON}"

uv run --extra llama python scripts/llama_train_qlora.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  --train-file "data/llama/llama_h${HORIZON}_train.csv" \
  --output-dir "results/generated/llama/h${HORIZON}_adapter"

uv run --extra llama python scripts/llama_eval.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --adapter-path "results/generated/llama/h${HORIZON}_adapter" \
  --test-file "data/llama/llama_h${HORIZON}_test.csv" \
  --out "results/generated/llama/h${HORIZON}_predictions.csv" \
  --compute-yes-no-probs

uv run python scripts/llama_threshold.py \
  --input "results/generated/llama/h${HORIZON}_predictions.csv" \
  --out "results/generated/llama/h${HORIZON}_thresholded.csv"
