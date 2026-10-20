# Source from the repository root on Athena.
export SCRATCH="${SCRATCH:-/net/tscratch/people/plgmarcinkostrzewa}"
export V4_ASSETS="${V4_ASSETS:-$SCRATCH/V4FinBench-assets}"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export HF_HOME="$V4_ASSETS/hf"
export HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="$V4_ASSETS/tabpfn"
export TABPFN_NO_BROWSER=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
