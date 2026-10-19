import argparse
import json
from pathlib import Path

import pandas as pd

from v4finbench.data.schema import COUNTRIES, COUNTRY_COL
from v4finbench.data.splits import TRACKS, in_track
from v4finbench.evaluation.summary import evaluate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-row test scores with firm-clustered bootstrap CIs."
    )
    parser.add_argument(
        "--results", type=Path, nargs="+", default=[Path("results/generated")]
    )
    parser.add_argument("--out", type=Path, default=Path("results/generated/summary"))
    parser.add_argument("--track", choices=TRACKS, default="main")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def check_same_tasks(scores: pd.DataFrame) -> None:
    tasks = scores.groupby("model")[["horizon", "test_year"]].apply(
        lambda frame: frozenset(map(tuple, frame.drop_duplicates().to_numpy()))
    )
    if tasks.nunique() > 1:
        raise ValueError(
            "Models cover different tasks; evaluate them separately or rerun: "
            + ", ".join(f"{model}: {len(t)} tasks" for model, t in tasks.items())
        )


def main() -> None:
    args = parse_args()
    roots = [root for root in args.results if root != args.out]
    scores = pd.concat(
        pd.read_parquet(path)
        for root in roots
        for path in sorted(root.rglob("*.parquet"))
    )
    scores = scores[
        [
            in_track(h, y, args.track)
            for h, y in zip(scores.horizon, scores.test_year, strict=True)
        ]
    ]
    check_same_tasks(scores)
    per_split = pd.DataFrame(
        json.loads(path.read_text(encoding="utf-8"))
        for root in roots
        for path in sorted(root.rglob("*.metrics.json"))
    )

    subsets = {
        "all": scores,
        "seen": scores[~scores["unseen_firm"]],
        "unseen": scores[scores["unseen_firm"]],
    } | {name: scores[scores[COUNTRY_COL] == code] for code, name in COUNTRIES.items()}
    summaries, paired = [], None
    for subset, rows in subsets.items():
        summary, subset_paired = evaluate_scores(
            rows, n_boot=args.n_boot, seed=args.seed
        )
        summaries.append(summary.assign(firms=subset))
        if subset == "all":
            paired = subset_paired
    summary = pd.concat(summaries, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / f"{args.track}_summary.csv", index=False)
    paired.to_csv(args.out / f"{args.track}_paired.csv", index=False)
    per_split.to_csv(args.out / "per_split.csv", index=False)
    columns = ["model", "horizon", "n_tasks", "roc_auc", "roc_auc_pooled"]
    columns += ["recall_at_fpr_0.01", "pr_auc", "f1"]
    print(summary[summary.firms == "all"][columns].round(3).to_string(index=False))
    print(
        f"Wrote {args.track}_summary.csv, {args.track}_paired.csv, per_split.csv "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
