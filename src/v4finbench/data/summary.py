from pathlib import Path

import pandas as pd

from v4finbench.data.schema import HORIZON_FILES, UNLABELED_FILE


def summarize_dataset_dir(
    data_dir: str | Path,
    target_col: str = "main_label",
) -> pd.DataFrame:
    root = Path(data_dir)
    rows = []

    unlabeled_path = root / UNLABELED_FILE
    if unlabeled_path.exists():
        df = pd.read_parquet(unlabeled_path, columns=["company"])
        rows.append(
            {
                "file": UNLABELED_FILE,
                "horizon": None,
                "n_rows": len(df),
                "n_companies": df["company"].nunique(),
                "positives": None,
                "positive_rate": None,
            }
        )

    for horizon, file_name in HORIZON_FILES.items():
        path = root / file_name
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["company", target_col])
        positives = int(df[target_col].sum())
        rows.append(
            {
                "file": file_name,
                "horizon": horizon,
                "n_rows": len(df),
                "n_companies": df["company"].nunique(),
                "positives": positives,
                "positive_rate": positives / len(df) if len(df) else 0.0,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No V4FinBench parquet files found in {root}")
    return pd.DataFrame(rows)
