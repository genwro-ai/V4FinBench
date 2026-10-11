#!/usr/bin/env bash
# One-time environment setup for the walk-forward TabPFN batch on Athena.
#
# Run this INSIDE an interactive job on a compute node (login nodes lack
# development libraries, and heavy non-GPU work on login nodes is against
# policy). From the repo root on $PLG_GROUPS_STORAGE:
#
#   srun -A plgfoundationeconom-gpu-a100 -p plgrid-gpu-a100 --gres=gpu:1 \
#        --cpus-per-task=8 --mem=64G -t 01:00:00 --pty bash -l
#   cd <repo>
#   bash slurm/setup_athena.sh
#
# Afterwards, submit the batch from the login node:
#   N=$(wc -l < data/temporal_splits/cells.txt)
#   MODEL_PATH=<tabpfn-checkpoint> sbatch --array=0-$((N-1)) \
#       slurm/tabpfn_temporal.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

source slurm/athena_env.sh
echo "uv cache:        ${UV_CACHE_DIR}"
echo "virtualenv:      ${UV_PROJECT_ENVIRONMENT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found - installing to ~/.local/bin (a few MB)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "=== Syncing environment (torch + tabpfn, first run downloads a lot) ==="
uv sync --extra tabpfn

echo "=== Sanity check: GPU visible to torch ==="
uv run python -c "import torch; print('cuda available:', torch.cuda.is_available())"

echo "=== Building walk-forward splits ==="
uv run python scripts/build_temporal_splits.py

mkdir -p logs

N=$(wc -l < data/temporal_splits/cells.txt)
echo ""
echo "=== Setup complete: ${N} cells ==="
echo "Submit with:"
echo "  MODEL_PATH=<tabpfn-checkpoint> sbatch --array=0-$((N-1)) slurm/tabpfn_temporal.sbatch"
