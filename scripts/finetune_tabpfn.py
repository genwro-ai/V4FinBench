import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from v4finbench.data.schema import HORIZON_FILES
from v4finbench.models.tabpfn_finetune import (
    finetune_config_from_mapping,
    finetune_evaluate_tabpfn,
    finetune_result_to_row,
    select_best_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TabPFN on V4FinBench.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tabpfn/finetune_prototype_undersample.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds-dir", type=Path, default=Path("data/folds"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/tabpfn_finetune"),
    )
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--target-col", default="main_label")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--prototype-backend",
        choices=["cuml", "sklearn"],
        default=None,
    )
    parser.add_argument("--prototype-device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = _load_config(args.config)
    overrides = {
        "model_path": args.model_path,
        "device": args.device,
        "prototype_backend": args.prototype_backend or args.prototype_device,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = finetune_config_from_mapping(values)

    data_path = args.data_dir / HORIZON_FILES[args.horizon]
    if not data_path.exists():
        raise FileNotFoundError(f"Missing horizon data file: {data_path}")
    print(f"Loading paper horizon h={args.horizon}: {data_path}", flush=True)
    df = pd.read_parquet(data_path)
    train_idx, val_idx, test_idx = _load_split_indices(
        args.folds_dir / f"h{args.horizon}",
        args.fold,
    )

    output_dir = args.out / f"h{args.horizon}" / f"fold_{args.fold}"
    _remove_if_exists(output_dir / "metrics.csv")
    _remove_if_exists(output_dir / "best_epoch.json")
    results = finetune_evaluate_tabpfn(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        horizon=args.horizon,
        fold=args.fold,
        config=config,
        target_col=args.target_col,
        output_dir=output_dir,
    )
    best = select_best_epoch(results)
    (output_dir / "metrics.json").write_text(
        json.dumps([finetune_result_to_row(result) for result in results], indent=2),
        encoding="utf-8",
    )
    print(
        f"Best epoch={best.epoch}: validation_f1={best.validation_f1:.4f}, "
        f"test_f1={best.metrics['f1']:.4f}, "
        f"test_roc_auc={best.metrics['roc_auc']:.4f}",
        flush=True,
    )


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_split_indices(
    fold_dir: Path,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.loadtxt(fold_dir / f"split_{fold}_train_idx.txt", dtype=int),
        np.loadtxt(fold_dir / f"split_{fold}_val_idx.txt", dtype=int),
        np.loadtxt(fold_dir / f"split_{fold}_test_idx.txt", dtype=int),
    )


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
