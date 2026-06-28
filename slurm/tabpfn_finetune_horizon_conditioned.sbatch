#!/usr/bin/env bash
#SBATCH --job-name=v4-tabpfn-ft-hc
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/athena_env.sh

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the TabPFN checkpoint path}"
FOLD="${SLURM_ARRAY_TASK_ID}"

uv run --extra tabpfn --extra rapids python scripts/finetune_tabpfn_horizon_conditioned.py \
  --config "${CONFIG:-configs/tabpfn/finetune_horizon_conditioned.yaml}" \
  --data-dir "${DATA_DIR:-data/raw}" \
  --folds-dir "${FOLDS_DIR:-data/folds}" \
  --out "${OUT_DIR:-results/generated/tabpfn_finetune_horizon_conditioned}" \
  --horizons "${HORIZONS:-all}" \
  --fold "${FOLD}" \
  --model-path "${MODEL_PATH}" \
  --device "${DEVICE:-cuda}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-8192}"
