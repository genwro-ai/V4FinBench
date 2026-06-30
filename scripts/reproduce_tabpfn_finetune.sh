#!/usr/bin/env bash
set -euo pipefail

if [[ -f scripts/athena_env.sh ]]; then
  source scripts/athena_env.sh
fi

DATA_DIR="${DATA_DIR:-data/raw}"
FOLDS_DIR="${FOLDS_DIR:-data/folds}"
OUT_DIR="${OUT_DIR:-results/generated/tabpfn_finetune}"
CONFIG="${CONFIG:-configs/tabpfn/finetune_prototype_undersample.yaml}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the TabPFN checkpoint path}"
DEVICE="${DEVICE:-cuda}"

for horizon in 0 1 2 3 4 5; do
  for fold in 0 1 2 3 4; do
    uv run --extra tabpfn --extra rapids python scripts/finetune_tabpfn.py \
      --config "${CONFIG}" \
      --data-dir "${DATA_DIR}" \
      --folds-dir "${FOLDS_DIR}" \
      --out "${OUT_DIR}" \
      --horizon "${horizon}" \
      --fold "${fold}" \
      --model-path "${MODEL_PATH}" \
      --device "${DEVICE}"
  done
done

uv run python scripts/aggregate_finetune_best.py \
  --root "${OUT_DIR}" \
  --out "${OUT_DIR}/best_epochs.csv" \
  --summary "${OUT_DIR}/summary.csv"
