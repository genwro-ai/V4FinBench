#!/bin/bash
#SBATCH --job-name=v4-llama-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

uv run python scripts/llama_prepare_data.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  --data-dir data/raw \
  --out data/llama \
  "$@"

