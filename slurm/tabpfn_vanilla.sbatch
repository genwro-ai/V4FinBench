#!/usr/bin/env bash
#SBATCH --job-name=v4-tabpfn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
uv run --extra tabpfn --extra rapids python scripts/run_tabpfn.py \
  --config "${CONFIG:-configs/tabpfn/vanilla_prototype_undersample.yaml}" \
  --data-dir "${DATA_DIR:-data/raw}" \
  --folds-dir "${FOLDS_DIR:-data/folds}" \
  --out "${OUT_DIR:-results/generated/tabpfn}"
