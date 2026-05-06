#!/usr/bin/env bash
#SBATCH --job-name=v4-baselines
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
uv run python scripts/run_baselines.py \
  --data-dir "${DATA_DIR:-data/raw}" \
  --folds-dir "${FOLDS_DIR:-data/folds}" \
  --out "${OUT_DIR:-results/generated/baselines}"
