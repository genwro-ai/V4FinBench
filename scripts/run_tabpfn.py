import argparse
from pathlib import Path
from typing import Any

import yaml

from v4finbench.data.io import iter_splits
from v4finbench.data.schema import HORIZONS
from v4finbench.data.splits import TRACKS
from v4finbench.evaluation.results import upsert_rows, write_test_scores
from v4finbench.models.tabpfn.vanilla import (
    TabPFNRunConfig,
    run_tabpfn,
    tabpfn_config_from_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vanilla (in-context) TabPFN on the walk-forward splits."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results/generated/tabpfn"))
    parser.add_argument(
        "--horizons", type=int, nargs="+", choices=HORIZONS, default=HORIZONS
    )
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--track", choices=TRACKS, default="full")
    parser.add_argument("--sampling-strategy", default=None)
    parser.add_argument("--n-context-samples", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--minority-to-majority-ratio", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_tabpfn_config(args)
    for df, split in iter_splits(
        args.data_dir, args.horizons, args.test_year, args.track
    ):
        cell = f"h{split.horizon}/Y{split.test_year}"
        result = run_tabpfn(df, split, config)
        upsert_rows(args.out / "metrics.csv", [result.to_row()])
        if not args.no_scores:
            path = args.out / "scores" / cell / f"{result.model}.parquet"
            write_test_scores(path, result, df)
        m = result.metrics
        print(
            f"{result.model} {cell}: ROC-AUC={m['roc_auc']:.3f} "
            f"PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f}",
            flush=True,
        )


def build_tabpfn_config(args: argparse.Namespace) -> TabPFNRunConfig:
    values = _load_config(args.config)
    overrides = {
        "sampling_strategy": args.sampling_strategy,
        "n_context_samples": args.n_context_samples,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "minority_to_majority_ratio": args.minority_to_majority_ratio,
        "random_state": args.random_state,
        "device": args.device,
        "model_path": args.model_path,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return tabpfn_config_from_mapping(values)


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if config.get("mode", "vanilla") != "vanilla":
        raise ValueError(f"{path} is not a vanilla TabPFN config.")
    return config


if __name__ == "__main__":
    main()
