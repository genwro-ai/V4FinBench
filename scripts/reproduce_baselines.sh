#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw}"
FOLDS_DIR="${FOLDS_DIR:-data/folds}"
OUT_DIR="${OUT_DIR:-results/generated/baselines}"
MAX_CANDIDATES="${MAX_CANDIDATES:-}"

ARGS=(
  --data-dir "${DATA_DIR}"
  --folds-dir "${FOLDS_DIR}"
  --out "${OUT_DIR}"
)

if [[ -n "${MAX_CANDIDATES}" ]]; then
  ARGS+=(--max-candidates "${MAX_CANDIDATES}")
fi

uv run python scripts/run_baselines.py "${ARGS[@]}"

uv run python scripts/aggregate_results.py \
  --input "${OUT_DIR}/metrics.csv" \
  --out "${OUT_DIR}/summary.csv"
