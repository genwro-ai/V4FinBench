#!/usr/bin/env bash
# Run once per checkout inside an interactive GPU allocation (srun --pty bash -l).
set -euo pipefail
source slurm/athena_env.sh

mkdir -p logs data/raw
[ -e data/raw/company_years.parquet ] || \
  ln -s "$V4_ASSETS/company_years.parquet" data/raw/company_years.parquet
uv sync --extra dev --extra neural --extra tabpfn
uv run --no-sync python scripts/build_labels.py
uv run --no-sync python scripts/build_temporal_splits.py
awk '$1 + $2 >= 2018 && $1 + $2 <= 2020' data/temporal_splits/cells.txt \
  > data/temporal_splits/cells_main.txt

uv run --no-sync python - <<'PY'
import numpy as np, torch
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else "")
X = np.random.default_rng(0).normal(size=(500, 8)).astype(np.float32)
y = (X[:, 0] > 0).astype(int)
for cls in (TabPFNClassifier, TabICLClassifier):
    print(cls.__name__, cls().fit(X, y).predict_proba(X[:2])[:, 1])
PY
echo "Main-track tasks: $(wc -l < data/temporal_splits/cells_main.txt)"
