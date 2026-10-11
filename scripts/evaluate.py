import argparse
import json
from pathlib import Path

import pandas as pd

from v4finbench.evaluation.summary import evaluate_scores

PRIMARY_TEST_YEARS = [2013, 2014, 2015]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-row test scores with firm-clustered bootstrap CIs."
    )
    parser.add_argument(
        "--results", type=Path, nargs="+", default=[Path("results/generated")]
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
    roots = [root for root in args.results if root != args.out]
    scores = pd.concat(
        pd.read_parquet(path)
        for root in roots
        for path in sorted(root.rglob("*.parquet"))
    )
    per_split = pd.DataFrame(
        json.loads(path.read_text(encoding="utf-8"))
        for root in roots
        for path in sorted(root.rglob("*.metrics.json"))
    )
    if not args.all_years:
        scores = scores[scores["test_year"].isin(PRIMARY_TEST_YEARS)]

    summaries, paired = [], None
    for subset, rows in {
        "all": scores,
        "seen": scores[~scores["unseen_firm"]],
        "unseen": scores[scores["unseen_firm"]],
    }.items():
        summary, subset_paired = evaluate_scores(
            rows, n_boot=args.n_boot, seed=args.seed
        )
        summaries.append(summary.assign(firms=subset))
        if subset == "all":
            paired = subset_paired
    summary = pd.concat(summaries, ignore_index=True)

    window = "all_years" if args.all_years else "primary"
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / f"{window}_summary.csv", index=False)
    paired.to_csv(args.out / f"{window}_paired.csv", index=False)
    per_split.to_csv(args.out / "per_split.csv", index=False)
    columns = ["firms", "model", "horizon", "roc_auc", "pr_auc", "recall_at_fpr_0.01"]
    print(summary[columns].round(3).to_string(index=False))
    print(
        f"Wrote {window}_summary.csv, {window}_paired.csv, per_split.csv to {args.out}"
    )


if __name__ == "__main__":
    main()
