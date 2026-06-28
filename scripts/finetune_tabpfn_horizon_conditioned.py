import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from v4finbench.data.horizon_joining import (
    JoinedHorizonSplitFrames,
    join_horizon_split_frames,
)
from v4finbench.data.schema import HORIZON_FILES
from v4finbench.models.tabpfn_finetune import (
    finetune_config_from_mapping,
    finetune_evaluate_tabpfn_splits,
    finetune_result_to_row,
    select_best_epoch,
)

JOINT_HORIZON_ID = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune one horizon-conditioned TabPFN model on joined "
            "V4FinBench horizon splits."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tabpfn/finetune_horizon_conditioned.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds-dir", type=Path, default=Path("data/folds"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/tabpfn_finetune_horizon_conditioned"),
    )
    parser.add_argument(
        "--horizons",
        default="all",
        help="'all', one horizon, or a comma-separated list such as 0,1,2.",
    )
    parser.add_argument(
        "--fold",
        default="all",
        help="'all', one fold, or a comma-separated list such as 0,1.",
    )
    parser.add_argument("--horizon-col", default="prediction_horizon")
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
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional joined train split subsample for local smoke testing.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional joined validation/test split subsample for local smoke testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = _parse_selection(args.horizons, sorted(HORIZON_FILES))
    folds = _parse_selection(args.fold, list(range(5)))
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

    data_by_horizon = _load_horizon_data(args.data_dir, horizons)
    horizon_dir = _horizon_dir_name(horizons)

    for fold in folds:
        splits_by_horizon = {
            horizon: _load_split_indices(args.folds_dir / f"h{horizon}", fold)
            for horizon in horizons
        }
        joined = join_horizon_split_frames(
            data_by_horizon=data_by_horizon,
            splits_by_horizon=splits_by_horizon,
            horizon_col=args.horizon_col,
        )
        output_dir = args.out / horizon_dir / f"fold_{fold}"
        _remove_if_exists(output_dir / "metrics.csv")
        _remove_if_exists(output_dir / "metrics.json")
        _remove_if_exists(output_dir / "best_epoch.json")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_join_metadata(output_dir, horizons, args.horizon_col, joined)

        print(
            f"Running horizon-conditioned TabPFN fold={fold}, "
            f"horizons={horizons}, "
            f"rows train/val/test="
            f"{len(joined.train)}/{len(joined.val)}/{len(joined.test)}, "
            f"sampling={config.sampling_strategy}",
            flush=True,
        )
        results = finetune_evaluate_tabpfn_splits(
            train_df=joined.train,
            val_df=joined.val,
            test_df=joined.test,
            horizon=JOINT_HORIZON_ID,
            fold=fold,
            config=config,
            target_col=args.target_col,
            output_dir=output_dir,
        )
        best = select_best_epoch(results)
        rows = [finetune_result_to_row(result) for result in results]
        (output_dir / "metrics.json").write_text(
            json.dumps(rows, indent=2),
            encoding="utf-8",
        )
        print(
            f"Best fold={fold} epoch={best.epoch}: "
            f"validation_f1={best.validation_f1:.4f}, "
            f"test_f1={best.metrics['f1']:.4f}, "
            f"test_roc_auc={best.metrics['roc_auc']:.4f}",
            flush=True,
        )


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    mode = values.get("mode", "finetune_horizon_conditioned")
    if mode not in {"finetune", "finetune_horizon_conditioned"}:
        raise ValueError(
            "finetune_tabpfn_horizon_conditioned.py expects a fine-tune config, "
            f"got mode={mode!r}"
        )
    return values


def _load_horizon_data(data_dir: Path, horizons: list[int]) -> dict[int, pd.DataFrame]:
    data_by_horizon = {}
    for horizon in horizons:
        data_path = data_dir / HORIZON_FILES[horizon]
        if not data_path.exists():
            raise FileNotFoundError(f"Missing horizon data file: {data_path}")
        print(f"Loading paper horizon h={horizon}: {data_path}", flush=True)
        data_by_horizon[horizon] = pd.read_parquet(data_path)
    return data_by_horizon


def _load_split_indices(
    fold_dir: Path,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _load_index_file(fold_dir / f"split_{fold}_train_idx.txt"),
        _load_index_file(fold_dir / f"split_{fold}_val_idx.txt"),
        _load_index_file(fold_dir / f"split_{fold}_test_idx.txt"),
    )


def _load_index_file(path: Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path, dtype=int)).astype(int)


def _parse_selection(value: str, allowed: list[int]) -> list[int]:
    if value == "all":
        return allowed
    selected = sorted({int(item) for item in value.split(",")})
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid selection {invalid}; allowed values: {allowed}")
    return selected


def _horizon_dir_name(horizons: list[int]) -> str:
    if horizons == sorted(HORIZON_FILES):
        return "h_all"
    return "h" + "-".join(str(horizon) for horizon in horizons)


def _write_join_metadata(
    output_dir: Path,
    horizons: list[int],
    horizon_col: str,
    joined: JoinedHorizonSplitFrames,
) -> None:
    metadata = {
        "horizons": horizons,
        "horizon_col": horizon_col,
        "joint_horizon_id": JOINT_HORIZON_ID,
        "split_rows": {
            "train": len(joined.train),
            "val": len(joined.val),
            "test": len(joined.test),
        },
    }
    (output_dir / "join_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
