import argparse
from pathlib import Path

import pandas as pd

from v4finbench.data.labels import (
    LabelConfig,
    write_horizon_labels,
)
from v4finbench.data.schema import UNLABELED_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate V4FinBench horizon label files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw") / UNLABELED_FILE,
        help="Path to company_years.parquet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed"),
        help="Directory for generated company_years_h*.parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing unlabeled input file: {args.input}")

    print(f"Loading {args.input} ...", flush=True)
    df = pd.read_parquet(args.input)
    config = LabelConfig()
    n_companies = df[config.group_col].nunique()
    print(
        f"Loaded {len(df):,} rows for {n_companies:,} companies. "
        "Generating 6 horizon files...",
        flush=True,
    )

    written = write_horizon_labels(
        df,
        args.out,
        config=config,
        show_progress=True,
    )
    for horizon, path in written.items():
        labeled = pd.read_parquet(path, columns=[config.target_col])
        positives = int(labeled[config.target_col].sum())
        print(
            f"Wrote paper horizon h={horizon}: {path} "
            f"({len(labeled):,} rows, {positives:,} positives)",
            flush=True,
        )


if __name__ == "__main__":
    main()
