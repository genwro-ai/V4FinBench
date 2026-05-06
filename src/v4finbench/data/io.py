from pathlib import Path

import pandas as pd

from v4finbench.data.schema import HORIZON_FILES, UNLABELED_FILE


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return path


def read_unlabeled(data_dir: str | Path) -> pd.DataFrame:
    path = require_file(Path(data_dir) / UNLABELED_FILE)
    return pd.read_parquet(path)


def read_horizon(data_dir: str | Path, horizon: int) -> pd.DataFrame:
    try:
        file_name = HORIZON_FILES[horizon]
    except KeyError as exc:
        raise ValueError(f"Unknown paper horizon: {horizon}") from exc
    return pd.read_parquet(require_file(Path(data_dir) / file_name))


def iter_horizon_files(data_dir: str | Path) -> list[tuple[int, Path]]:
    root = Path(data_dir)
    return [
        (horizon, require_file(root / file_name))
        for horizon, file_name in HORIZON_FILES.items()
    ]
