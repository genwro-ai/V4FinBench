from __future__ import annotations

import argparse
from pathlib import Path

from v4finbench.evaluation.results import aggregate_best_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate TabPFN fine-tuning best-epoch files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/generated/tabpfn_finetune"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/tabpfn_finetune/best_epochs.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/generated/tabpfn_finetune/summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_epochs, summary = aggregate_best_epochs(args.root, args.out, args.summary)
    print(
        f"Wrote {len(best_epochs):,} best-epoch rows to {args.out} "
        f"and {len(summary):,} summary rows to {args.summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
