#!/usr/bin/env bash
# One array element = one (horizon, test year) line of TASKS.
#   RUNNER=baseline CONFIG=configs/foundation/in_context.yaml MODEL=tabpfn_sample \
#     sbatch --array=0-14 slurm/task.sbatch
#   RUNNER=finetune ESTIMATOR=tabpfn SAMPLING=none sbatch --array=0-14 slurm/task.sbatch
#SBATCH --job-name=v4finbench
#SBATCH --account=plgfoundationeconom-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/athena_env.sh

TASKS="${TASKS:-data/temporal_splits/cells_main.txt}"
read -r HORIZON TEST_YEAR < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
echo "commit=$(git rev-parse --short HEAD) runner=$RUNNER horizon=$HORIZON test_year=$TEST_YEAR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "$RUNNER" in
  baseline)
    uv run --no-sync python scripts/run_baselines.py \
      --config "$CONFIG" --models "$MODEL" \
      --horizons "$HORIZON" --test-year "$TEST_YEAR" \
      --out "${OUT:-results/generated/baselines}" --no-progress ;;
  finetune)
    uv run --no-sync python scripts/finetune_foundation.py \
      --estimator "$ESTIMATOR" --sampling "$SAMPLING" \
      --horizons "$HORIZON" --test-year "$TEST_YEAR" \
      --out "${OUT:-results/generated/finetune}" ;;
  *) echo "Unknown RUNNER=$RUNNER" >&2; exit 1 ;;
esac
