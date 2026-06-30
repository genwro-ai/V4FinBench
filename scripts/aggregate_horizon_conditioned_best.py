import argparse
from pathlib import Path

from v4finbench.evaluation.results import aggregate_best_epochs_by_horizon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate horizon-conditioned TabPFN per-horizon best-epoch metrics."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/generated/tabpfn_finetune_horizon_conditioned"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/generated/tabpfn_finetune_horizon_conditioned/"
            "best_epochs_by_horizon.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/generated/tabpfn_finetune_horizon_conditioned/"
            "summary_by_horizon.csv"
        ),
    )
    parser.add_argument(
        "--means",
        type=Path,
        default=Path(
            "results/generated/tabpfn_finetune_horizon_conditioned/"
            "mean_metrics_by_horizon.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_epochs, summary, means = aggregate_best_epochs_by_horizon(
        args.root,
        args.out,
        args.summary,
        args.means,
    )
    print(
        f"Wrote {len(best_epochs):,} best-epoch horizon rows to {args.out}, "
        f"{len(summary):,} summary rows to {args.summary}, "
        f"and {len(means):,} mean rows to {args.means}",
        flush=True,
    )


if __name__ == "__main__":
    main()
