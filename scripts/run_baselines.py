import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from v4finbench.data.schema import HORIZON_FILES
from v4finbench.models.baselines import (
    BASELINE_MODEL_NAMES,
    BaselineRunResult,
    train_evaluate_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4FinBench standard baselines.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds-dir", type=Path, default=Path("data/folds"))
    parser.add_argument("--out", type=Path, default=Path("results/generated/baselines"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/standard.yaml"),
    )
    parser.add_argument("--horizon", default="all", help="'all' or paper horizon 0..5")
    parser.add_argument("--fold", default="all", help="'all' or fold index")
    parser.add_argument("--model", default="all", help="'all' or one baseline model")
    parser.add_argument("--target-col", default="main_label")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Randomly sample at most this many hyperparameter candidates per run.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Do not save fitted model artifacts.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing metrics files instead of replacing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_config = _load_baseline_config(args.config)
    horizons = _parse_selection(args.horizon, sorted(HORIZON_FILES))
    folds = _parse_selection(args.fold, list(range(5)))
    models = BASELINE_MODEL_NAMES if args.model == "all" else [args.model]

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.out / "metrics.csv"
    metrics_json = args.out / "metrics.json"
    if not args.append:
        _remove_if_exists(metrics_csv)
        _remove_if_exists(metrics_json)

    rows = []
    for horizon in horizons:
        data_path = args.data_dir / HORIZON_FILES[horizon]
        if not data_path.exists():
            raise FileNotFoundError(f"Missing horizon data file: {data_path}")
        print(f"Loading paper horizon h={horizon}: {data_path}", flush=True)
        df = pd.read_parquet(data_path)

        for fold in folds:
            train_idx, val_idx, test_idx = _load_split_indices(
                args.folds_dir / f"h{horizon}",
                fold,
            )
            for model_name in models:
                if model_name not in baseline_config["models"]:
                    raise ValueError(
                        f"Model {model_name} is not present in {args.config}"
                    )
                print(
                    f"Running model={model_name}, h={horizon}, fold={fold}",
                    flush=True,
                )
                artifact_dir = None
                if not args.no_save_model:
                    artifact_dir = args.out / f"h{horizon}" / f"fold_{fold}"
                result = train_evaluate_baseline(
                    df=df,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    model_name=model_name,
                    param_grid=baseline_config["models"][model_name],
                    horizon=horizon,
                    fold=fold,
                    target_col=args.target_col,
                    random_state=args.random_state,
                    max_candidates=args.max_candidates,
                    output_dir=artifact_dir,
                )
                row = _result_to_row(result)
                rows.append(row)
                _append_results(metrics_csv, [row])
                print(
                    f"Finished model={model_name}, h={horizon}, fold={fold}: "
                    f"f1={result.metrics['f1']:.4f}, "
                    f"roc_auc={result.metrics['roc_auc']:.4f}",
                    flush=True,
                )

    _write_results(metrics_json, rows)


def _load_baseline_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _parse_selection(value: str, allowed: list[int]) -> list[int]:
    if value == "all":
        return allowed
    selected = int(value)
    if selected not in allowed:
        raise ValueError(f"Invalid selection {value}; allowed values: {allowed}")
    return [selected]


def _load_split_indices(
    fold_dir: Path,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.loadtxt(fold_dir / f"split_{fold}_train_idx.txt", dtype=int),
        np.loadtxt(fold_dir / f"split_{fold}_val_idx.txt", dtype=int),
        np.loadtxt(fold_dir / f"split_{fold}_test_idx.txt", dtype=int),
    )


def _result_to_row(result: BaselineRunResult) -> dict[str, Any]:
    row = {
        "model": result.model_name,
        "horizon": result.horizon,
        "fold": result.fold,
        "validation_f1": result.validation_f1,
        "best_params": json.dumps(result.best_params, sort_keys=True),
    }
    row.update(result.metrics)
    return row


def _append_results(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
