import argparse
from pathlib import Path

import pandas as pd

from v4finbench.data.folds import FoldConfig, write_split_files
from v4finbench.data.schema import HORIZON_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic V4FinBench fold files."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/folds"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--group-col", default="company")
    parser.add_argument("--target-col", default="main_label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FoldConfig(
        n_splits=args.n_splits,
        random_state=args.seed,
        country_col=args.country_col,
        group_col=args.group_col,
        target_col=args.target_col,
    )

    for horizon, file_name in HORIZON_FILES.items():
        path = args.data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing horizon file: {path}")

        df = pd.read_parquet(path)
        dataset_out = args.out / f"h{horizon}"
        write_split_files(df, dataset_out, config)
        print(f"Wrote folds for paper horizon h={horizon} to {dataset_out}")


if __name__ == "__main__":
    main()
