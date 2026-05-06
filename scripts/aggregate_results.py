import argparse
from pathlib import Path

from v4finbench.evaluation.results import aggregate_metrics_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate V4FinBench result metrics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/generated/baselines/metrics.csv"),
        help="Input per-run metrics CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/baselines/summary.csv"),
        help="Output summary CSV.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["model", "horizon"],
        help="Columns used to group results before aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_metrics_file(args.input, args.out, group_columns=args.group_by)
    print(f"Wrote {len(summary):,} summary rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
