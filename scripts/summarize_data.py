import argparse
from pathlib import Path

import pandas as pd

from v4finbench.data.io import read_horizon
from v4finbench.data.schema import COUNTRIES, FIRM_COL, HORIZONS, TARGET_COL, YEAR_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the horizon files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/data_summary")
    )
    return parser.parse_args()


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    summary = df.groupby(by).agg(
        n_rows=(TARGET_COL, "size"),
        n_firms=(FIRM_COL, "nunique"),
        positives=(TARGET_COL, "sum"),
    )
    summary["positive_rate"] = summary["positives"] / summary["n_rows"]
    return summary.reset_index()


def main() -> None:
    args = parse_args()
    columns = [FIRM_COL, YEAR_COL, "country", TARGET_COL]
    labeled = pd.concat(
        read_horizon(args.data_dir, horizon)[columns].assign(horizon=horizon)
        for horizon in HORIZONS
    )
    labeled["country"] = labeled["country"].map(COUNTRIES)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, by in {
        "by_horizon": ["horizon"],
        "by_horizon_country": ["horizon", "country"],
        "by_horizon_year": ["horizon", YEAR_COL],
    }.items():
        summary = summarize(labeled, by)
        summary.to_csv(args.out / f"{name}.csv", index=False)
        if name == "by_horizon":
            print(summary.to_string(index=False))
    print(f"Wrote summaries to {args.out}")


if __name__ == "__main__":
    main()
