import argparse
from pathlib import Path

import pandas as pd

from v4finbench.evaluation.summary import evaluate_scores

PRIMARY_TEST_YEARS = [2013, 2014, 2015]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-row test scores with firm-clustered bootstrap CIs."
    )
    parser.add_argument(
        "--scores", type=Path, nargs="+", default=[Path("results/generated")]
    )
    parser.add_argument("--out", type=Path, default=Path("results/generated/summary"))
    parser.add_argument(
        "--all-years",
        action="store_true",
        help=f"Use every available test year instead of {PRIMARY_TEST_YEARS}.",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(path for root in args.scores for path in root.rglob("*.parquet"))
    scores = pd.concat(pd.read_parquet(path) for path in files)
    if not args.all_years:
        scores = scores[scores["test_year"].isin(PRIMARY_TEST_YEARS)]
    summary, paired = evaluate_scores(scores, n_boot=args.n_boot, seed=args.seed)

    window = "all_years" if args.all_years else "primary"
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / f"{window}_summary.csv", index=False)
    paired.to_csv(args.out / f"{window}_paired.csv", index=False)
    columns = ["model", "horizon", "roc_auc", "pr_auc", "recall_at_fpr_0.01", "f1"]
    print(summary[columns].round(3).to_string(index=False))
    print(f"Wrote {window}_summary.csv and {window}_paired.csv to {args.out}")


if __name__ == "__main__":
    main()
