#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw}"
FOLDS_DIR="${FOLDS_DIR:-data/folds}"
OUT_DIR="${OUT_DIR:-results/generated/tabpfn}"
CONFIG="${CONFIG:-configs/tabpfn/vanilla_prototype_undersample.yaml}"

uv run --extra tabpfn python scripts/run_tabpfn.py \
  --config "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --folds-dir "${FOLDS_DIR}" \
  --out "${OUT_DIR}"

uv run python scripts/aggregate_results.py \
  --input "${OUT_DIR}/metrics.csv" \
  --out "${OUT_DIR}/summary.csv" \
  --group-by model horizon sampling_strategy
