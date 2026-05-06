#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
FOLDS_DIR="${FOLDS_DIR:-data/folds}"
RESULTS_DIR="${RESULTS_DIR:-results/generated}"

uv run python scripts/build_labels.py \
  --input "${DATA_DIR}/company_years.parquet" \
  --out "${PROCESSED_DIR}"

uv run python scripts/verify_labels.py \
  --generated "${PROCESSED_DIR}" \
  --reference "${DATA_DIR}"

uv run python scripts/summarize_data.py \
  --data-dir "${DATA_DIR}" \
  --out "${RESULTS_DIR}/data_summary.csv"

uv run python scripts/build_folds.py \
  --data-dir "${DATA_DIR}" \
  --out "${FOLDS_DIR}" \
  --seed 42
