import argparse
from pathlib import Path

import pandas as pd

from v4finbench.data.labels import create_horizon_labels
from v4finbench.data.schema import FIRM_COL, HORIZONS, TARGET_COL, horizon_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the horizon label files.")
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/company_years.parquet")
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--horizons", type=int, nargs="+", choices=HORIZONS, default=HORIZONS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    print(f"{len(df):,} rows, {df[FIRM_COL].nunique():,} firms")
    args.out.mkdir(parents=True, exist_ok=True)
    for horizon in args.horizons:
        labeled = create_horizon_labels(df, horizon)
        path = args.out / horizon_file(horizon)
        labeled.to_parquet(path, index=False)
        print(
            f"h={horizon}: {len(labeled):,} rows, "
            f"{int(labeled[TARGET_COL].sum()):,} positives -> {path}"
        )


if __name__ == "__main__":
    main()
