# Shared Athena environment for V4FinBench. Source this from setup scripts
# and sbatch jobs so both resolve the same caches and virtual environment.
#
# $HOME on Athena has a 10 GB quota: every cache lives in $SCRATCH instead.
# $SCRATCH is auto-cleaned after 30 days, which is fine for caches and the
# venv (one `bash slurm/setup_athena.sh` rebuilds them). Keep the repo, the
# data, and results on $PLG_GROUPS_STORAGE, which persists.

export UV_CACHE_DIR="${SCRATCH}/v4finbench/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH}/v4finbench/uv-python"
export UV_PROJECT_ENVIRONMENT="${SCRATCH}/v4finbench/venv"
export HF_HOME="${SCRATCH}/v4finbench/hf-cache"
export PIP_CACHE_DIR="${SCRATCH}/v4finbench/pip-cache"
export TMPDIR="${SCRATCH}/v4finbench/tmp"

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" \
         "${UV_PROJECT_ENVIRONMENT%/*}" "${HF_HOME}" \
         "${PIP_CACHE_DIR}" "${TMPDIR}"
