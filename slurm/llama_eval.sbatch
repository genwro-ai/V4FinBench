#!/bin/bash
#SBATCH --job-name=v4-llama-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

uv run --extra llama python scripts/llama_eval.py "$@"

