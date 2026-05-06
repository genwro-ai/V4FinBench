from __future__ import annotations

import argparse
from pathlib import Path

from v4finbench.data.summary import summarize_dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V4FinBench parquet files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--target-col", default="main_label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_dataset_dir(args.data_dir, target_col=args.target_col)
    print(summary.to_string(index=False), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"Wrote summary to {args.out}", flush=True)


if __name__ == "__main__":
    main()

