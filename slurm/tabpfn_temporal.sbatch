#!/usr/bin/env bash
# TabPFN finetuning on the walk-forward (out-of-time) splits.
#
# The cell grid is triangular (long horizons lose recent test years, and
# thin cells are dropped by minimum-positive thresholds), so cells are read
# from the manifest written by the split builder:
#     data/temporal_splits/cells.txt   ("horizon origin_index test_year")
#
# Before first use: run slurm/setup_athena.sh inside an interactive compute
# node job (it creates the venv and caches in $SCRATCH and builds the
# splits - see its header for the srun command). Then size the array to the
# manifest:
#     N=$(wc -l < data/temporal_splits/cells.txt)
#     MODEL_PATH=<ckpt> sbatch --array=0-$((N-1)) slurm/tabpfn_temporal.sbatch
#
# To run only the full-grid test years (e.g. for the primary table), filter
# the manifest into a copy and point CELLS_FILE at it:
#     awk '$3 >= 2013 && $3 <= 2015' data/temporal_splits/cells.txt > /tmp/cells_primary.txt
#     N=$(wc -l < /tmp/cells_primary.txt)
#     MODEL_PATH=<ckpt> CELLS_FILE=/tmp/cells_primary.txt sbatch --array=0-$((N-1)) ...
#
# Smoke test on a dev node (no GPU queue, ~minutes):
#   uv run --extra tabpfn python scripts/finetune_tabpfn.py \
#     --config configs/tabpfn/finetune_prototype_undersample.yaml \
#     --data-dir data/processed --folds-dir data/temporal_splits \
#     --out results/generated/tabpfn_temporal_smoke \
#     --horizon 5 --fold 3 --drop-cols year \
#     --device cpu --epochs 1 --max-train-samples 4096 --max-eval-samples 256
#
# Athena (Cyfronet) specifics: partition plgrid-gpu-a100 (48 h walltime cap),
# proportional resources are 16 CPU cores and 128 GB RAM per A100. Compute
# nodes have no internet access: run `uv sync --extra tabpfn` on the login
# node BEFORE submitting, and MODEL_PATH must be a local checkpoint file.
#
#SBATCH --job-name=v4-tabpfn-oot
#SBATCH -A plgfoundationeconom-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Same caches and virtualenv as slurm/setup_athena.sh ($HOME quota is 10 GB,
# so everything lives in $SCRATCH).
source slurm/athena_env.sh

# The environment is fully synced by setup_athena.sh - do not resolve or
# download anything at job time.
export UV_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the TabPFN checkpoint path}"
CELLS_FILE="${CELLS_FILE:-data/temporal_splits/cells.txt}"

if [[ ! -f "${CELLS_FILE}" ]]; then
  echo "Missing ${CELLS_FILE} - run scripts/build_temporal_splits.py first." >&2
  exit 1
fi

N_CELLS=$(wc -l < "${CELLS_FILE}")
if (( SLURM_ARRAY_TASK_ID >= N_CELLS )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds manifest (${N_CELLS} cells)." >&2
  exit 1
fi

read -r HORIZON ORIGIN TEST_YEAR <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CELLS_FILE}")"
echo "Cell: horizon=${HORIZON} origin=${ORIGIN} test_year=${TEST_YEAR}"

uv run --extra tabpfn python scripts/finetune_tabpfn.py \
  --config "${CONFIG:-configs/tabpfn/finetune_prototype_undersample.yaml}" \
  --data-dir "${DATA_DIR:-data/processed}" \
  --folds-dir "${FOLDS_DIR:-data/temporal_splits}" \
  --out "${OUT_DIR:-results/generated/tabpfn_temporal}" \
  --horizon "${HORIZON}" \
  --fold "${ORIGIN}" \
  --drop-cols year \
  --model-path "${MODEL_PATH}" \
  --device "${DEVICE:-cuda}"
