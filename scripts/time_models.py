import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from v4finbench.data.io import iter_splits
from v4finbench.data.preprocessing import prepare_split
from v4finbench.data.schema import HORIZONS
from v4finbench.data.splits import TRACKS
from v4finbench.evaluation.results import hardware
from v4finbench.models.baselines import fit_model, positive_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit each task's selected configuration and time training "
        "and inference on this machine."
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baselines/standard.yaml")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--results", type=Path, default=Path("results/generated/baselines")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/summary/timing.csv")
    )
    parser.add_argument("--track", choices=TRACKS, default="main")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def timed(fn, *args):
    start = time.perf_counter()
    value = fn(*args)
    return value, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    rows = []
    for df, split in iter_splits(args.data_dir, list(HORIZONS), track=args.track):
        data = prepare_split(df, split.train_idx, split.val_idx, split.test_idx)
        cell = args.results / f"h{split.horizon}" / f"Y{split.test_year}"
        for model_name in args.models:
            run = json.loads((cell / f"{model_name}.metrics.json").read_text())
            params = json.loads(run["best_params"])
            estimator = config["models"][model_name].get("estimator", model_name)
            model, train_seconds = timed(
                fit_model,
                estimator,
                params,
                data,
                args.random_state,
                config.get("early_stopping_rounds"),
            )
            _, val_seconds = timed(positive_score, model, data.X_val)
            _, test_seconds = timed(positive_score, model, data.X_test)
            rows.append(
                {
                    "model": model_name,
                    "horizon": split.horizon,
                    "test_year": split.test_year,
                    "train_rows": len(data.y_train),
                    "test_rows": len(data.y_test),
                    "train_seconds": train_seconds,
                    "val_inference_seconds": val_seconds,
                    "test_inference_seconds": test_seconds,
                    "search_seconds": run.get("fit_seconds"),
                    **hardware(),
                }
            )
            print(
                f"{model_name:>20s} h{split.horizon} Y{split.test_year}: "
                f"train {train_seconds:.1f}s, test {test_seconds:.3f}s",
                flush=True,
            )

    timing = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    timing.to_csv(args.out, index=False)
    summary = timing.groupby("model").agg(
        train_seconds=("train_seconds", "mean"),
        test_inference_seconds=("test_inference_seconds", "mean"),
        test_rows=("test_rows", "mean"),
        search_minutes=("search_seconds", lambda s: s.mean() / 60),
    )
    summary["test_rows_per_second"] = (
        summary["test_rows"] / summary["test_inference_seconds"]
    )
    print(summary.round(3).to_string())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
