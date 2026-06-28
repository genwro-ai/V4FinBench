#!/usr/bin/env bash

# Source this file on Athena before running uv/TabPFN commands:
#   source scripts/athena_env.sh
#
# It keeps uv-managed Python, caches, and virtualenvs off the 10GB home quota
# and exposes CUDA libraries shipped by Python wheels, including NVRTC.

if [[ -n "${V4FINBENCH_SCRATCH_ROOT:-}" ]]; then
  scratch_root="${V4FINBENCH_SCRATCH_ROOT}"
elif [[ -n "${SCRATCH:-}" ]]; then
  scratch_root="${SCRATCH}"
else
  scratch_root="/net/tscratch/people/${USER}"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${scratch_root}/.cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${scratch_root}/.venvs/V4FinBench}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${scratch_root}/.local/share/uv/python}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-${scratch_root}/.local/share/uv/tools}"
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-${scratch_root}/.local/bin}"
export PATH="${UV_INSTALL_DIR}:${PATH}"

mkdir -p \
  "${UV_CACHE_DIR}" \
  "${UV_PROJECT_ENVIRONMENT%/*}" \
  "${UV_PYTHON_INSTALL_DIR}" \
  "${UV_TOOL_DIR}" \
  "${UV_INSTALL_DIR}"

if [[ -d "${UV_PROJECT_ENVIRONMENT}" ]]; then
  site_packages="$(
    find "${UV_PROJECT_ENVIRONMENT}/lib" \
      -maxdepth 3 \
      -type d \
      -name site-packages \
      2>/dev/null \
      | head -n 1
  )"

  if [[ -n "${site_packages}" ]]; then
    nvidia_libs="$(
      find "${site_packages}/nvidia" \
        -mindepth 2 \
        -maxdepth 3 \
        -type d \
        -name lib \
        2>/dev/null \
        | paste -sd:
    )"

    if [[ -n "${nvidia_libs}" ]]; then
      export LD_LIBRARY_PATH="${nvidia_libs}:${LD_LIBRARY_PATH:-}"
    fi
  fi
fi
