#!/usr/bin/env bash
#SBATCH --job-name=v4-tabpfn-ft
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the TabPFN checkpoint path}"
HORIZON=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))

uv run --extra tabpfn python scripts/finetune_tabpfn.py \
  --config "${CONFIG:-configs/tabpfn/finetune_prototype_undersample.yaml}" \
  --data-dir "${DATA_DIR:-data/raw}" \
  --folds-dir "${FOLDS_DIR:-data/folds}" \
  --out "${OUT_DIR:-results/generated/tabpfn_finetune}" \
  --horizon "${HORIZON}" \
  --fold "${FOLD}" \
  --model-path "${MODEL_PATH}" \
  --device "${DEVICE:-cuda}"
