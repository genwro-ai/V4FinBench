#!/usr/bin/env bash
# Llama inference-cost benchmark on Athena (Cyfronet).
#
# The benchmark uses the prepared V4FinBench chat prompts and runs:
#   - an untimed 32-sample warm-up,
#   - one timed 20,000-sample run,
#   - five timed repetitions on a fixed 5,000-sample subset.
#
# One-time preparation, before submission:
#
#   source slurm/athena_env.sh
#   uv sync --extra tabpfn --extra llama
#   uv run python scripts/llama_prepare_data.py \
#     --config configs/llama/qlora_llama3_8b.yaml \
#     --data-dir data/raw --out data/llama --horizon 0
#
# Meta-Llama-3-8B is gated. Accept its license on Hugging Face, authenticate,
# and download the base checkpoint while internet access is available:
#
#   MODEL_DIR="${SCRATCH}/v4finbench/models/Meta-Llama-3-8B"
#   uv run --extra llama hf download meta-llama/Meta-Llama-3-8B \
#     --local-dir "${MODEL_DIR}"
#
# Do not put an access token in this file or in the sbatch command. Use
# `hf auth login` or the HF_TOKEN environment variable during the download.
#
# Submit from the repository root after creating the log directory:
#
#   mkdir -p logs
#   MODEL_PATH="${SCRATCH}/v4finbench/models/Meta-Llama-3-8B" \
#     sbatch slurm/llama_inference_benchmark.sbatch
#
# Optional environment overrides:
#   TEST_FILE, OUT_FILE, ADAPTER_PATH, BATCH_SIZE, FULL_SAMPLES,
#   REPEAT_SAMPLES, REPETITIONS, WARMUP_SAMPLES, MAX_NEW_TOKENS, USE_4BIT.
#
# Athena A100 resources follow the same account and partition as the temporal
# TabPFN jobs. Compute nodes are treated as offline.
#
#SBATCH --job-name=v4-llama-cost
#SBATCH -A plgfoundationeconom-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source slurm/athena_env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV="${VENV:-/net/tscratch/people/plgmarcinkostrzewa/.venvs/V4FinBench}"
if [[ -f "${VENV}/bin/activate" ]]; then
  echo "Using existing venv: ${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  RUN=(python)
else
  echo "VENV not found, using uv environment from slurm/athena_env.sh"
  export UV_OFFLINE=1
  RUN=(uv run --no-sync --extra llama python)
fi
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

if ! "${RUN[@]}" -c "import accelerate, bitsandbytes, peft, torch, transformers"; then
  echo "The selected environment does not contain the Llama dependencies." >&2
  echo "Run: source slurm/athena_env.sh && uv sync --extra tabpfn --extra llama" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-${SCRATCH}/v4finbench/models/Meta-Llama-3-8B}"
TEST_FILE="${TEST_FILE:-data/llama/llama_h0_test.csv}"
OUT_FILE="${OUT_FILE:-results/generated/llama/inference_benchmark.json}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Local Llama checkpoint not found at MODEL_PATH=${MODEL_PATH}" >&2
  echo "Download the gated base weights before submitting the job." >&2
  exit 1
fi
if [[ ! -f "${TEST_FILE}" ]]; then
  echo "Prepared prompt file not found: ${TEST_FILE}" >&2
  echo "Run scripts/llama_prepare_data.py before submitting the job." >&2
  exit 1
fi

COMMAND=(
  "${RUN[@]}" scripts/benchmark_llama_inference.py
  --model-name "${MODEL_PATH}"
  --test-file "${TEST_FILE}"
  --out "${OUT_FILE}"
  --full-samples "${FULL_SAMPLES:-20000}"
  --repeat-samples "${REPEAT_SAMPLES:-5000}"
  --repetitions "${REPETITIONS:-5}"
  --warmup-samples "${WARMUP_SAMPLES:-32}"
  --batch-size "${BATCH_SIZE:-1}"
  --max-new-tokens "${MAX_NEW_TOKENS:-5}"
)

if [[ -n "${ADAPTER_PATH:-}" ]]; then
  if [[ ! -d "${ADAPTER_PATH}" ]]; then
    echo "ADAPTER_PATH does not exist: ${ADAPTER_PATH}" >&2
    exit 1
  fi
  COMMAND+=(--adapter-path "${ADAPTER_PATH}")
fi
if [[ "${USE_4BIT:-1}" == "0" ]]; then
  COMMAND+=(--no-4bit)
fi
COMMAND+=("$@")

echo "Model: ${MODEL_PATH}"
echo "Prompts: ${TEST_FILE}"
echo "Output: ${OUT_FILE}"
echo "SLURM job: ${SLURM_JOB_ID}"
nvidia-smi

"${COMMAND[@]}"
