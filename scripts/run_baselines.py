import argparse
from pathlib import Path

import yaml

from v4finbench.data.io import iter_splits
from v4finbench.data.schema import HORIZONS
from v4finbench.evaluation.results import upsert_rows, write_test_scores
from v4finbench.models.baselines import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tabular baselines on the walk-forward splits."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results/generated/baselines"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baselines/standard.yaml")
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="Default: every model in the config."
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", choices=HORIZONS, default=HORIZONS
    )
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-scores", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    models = args.models or list(config["models"])
    unknown = set(models) - set(config["models"])
    if unknown:
        raise ValueError(f"Models missing from {args.config}: {sorted(unknown)}")

    for df, split in iter_splits(args.data_dir, args.horizons, args.test_year):
        cell = f"h{split.horizon}/Y{split.test_year}"
        for model_name in models:
            result = run_baseline(
                df,
                split,
                model_name,
                config,
                random_state=args.random_state,
                n_trials=args.n_trials,
                output_dir=args.out / "models" / cell if args.save_models else None,
                progress=not args.no_progress,
            )
            upsert_rows(args.out / "metrics.csv", [result.to_row()])
            if not args.no_scores:
                path = args.out / "scores" / cell / f"{model_name}.parquet"
                write_test_scores(path, result, df)
            m = result.metrics
            print(
                f"{model_name:>20s} {cell}: ROC-AUC={m['roc_auc']:.3f} "
                f"PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f} "
                f"R@FPR1%={m['recall_at_fpr_0.01']:.3f}",
                flush=True,
            )
    print(f"Wrote {args.out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
