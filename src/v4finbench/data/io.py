from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from v4finbench.data.schema import horizon_file
from v4finbench.data.splits import TemporalSplit, temporal_splits


def read_horizon(data_dir: str | Path, horizon: int) -> pd.DataFrame:
    path = Path(data_dir) / horizon_file(horizon)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_labels.py")
    return pd.read_parquet(path)


def iter_splits(
    data_dir: str | Path,
    horizons: list[int],
    test_year: int | None = None,
) -> Iterator[tuple[pd.DataFrame, TemporalSplit]]:
    for horizon in horizons:
        df = read_horizon(data_dir, horizon)
        for split in temporal_splits(df, horizon):
            if test_year is None or split.test_year == test_year:
                yield df, split
