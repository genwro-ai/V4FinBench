import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from v4finbench.data.schema import HORIZON_FILES
from v4finbench.models.tabpfn import (
    TabPFNRunConfig,
    tabpfn_config_from_mapping,
    tabpfn_result_to_row,
    train_evaluate_tabpfn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4FinBench TabPFN evaluation.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds-dir", type=Path, default=Path("data/folds"))
    parser.add_argument("--out", type=Path, default=Path("results/generated/tabpfn"))
    parser.add_argument("--horizon", default="all", help="'all' or paper horizon 0..5")
    parser.add_argument("--fold", default="all", help="'all' or fold index")
    parser.add_argument("--target-col", default="main_label")
    parser.add_argument("--sampling-strategy", default=None)
    parser.add_argument("--n-context-samples", type=int, default=None)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional train split subsample for local smoke testing.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional validation/test split subsample for local smoke testing.",
    )
    parser.add_argument("--minority-to-majority-ratio", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional TabPFN checkpoint/weights path passed to TabPFNClassifier.",
    )
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = _parse_selection(args.horizon, sorted(HORIZON_FILES))
    folds = _parse_selection(args.fold, list(range(5)))
    config = build_tabpfn_config(args)

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
            print(
                f"Running TabPFN h={horizon}, fold={fold}, "
                f"sampling={config.sampling_strategy}, "
                f"context={config.n_context_samples}",
                flush=True,
            )
            result = train_evaluate_tabpfn(
                df=df,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                horizon=horizon,
                fold=fold,
                config=config,
                target_col=args.target_col,
            )
            row = tabpfn_result_to_row(result)
            rows.append(row)
            _append_results(metrics_csv, [row])
            print(
                f"Finished TabPFN h={horizon}, fold={fold}: "
                f"f1={result.metrics['f1']:.4f}, "
                f"roc_auc={result.metrics['roc_auc']:.4f}",
                flush=True,
            )

    metrics_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def build_tabpfn_config(args: argparse.Namespace) -> TabPFNRunConfig:
    values = _load_tabpfn_config_values(args.config)
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


def _load_tabpfn_config_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    mode = config.get("mode", "vanilla")
    if mode != "vanilla":
        raise ValueError(
            f"run_tabpfn.py currently supports only vanilla configs, got mode={mode}"
        )
    return config


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


def _append_results(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
