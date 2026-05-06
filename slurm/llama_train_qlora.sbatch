#!/bin/bash
#SBATCH --job-name=v4-llama-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

uv run --extra llama python scripts/llama_train_qlora.py \
  --config configs/llama/qlora_llama3_8b.yaml \
  "$@"

