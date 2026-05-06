import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldConfig:
    n_splits: int = 5
    random_state: int = 42
    country_col: str = "country"
    group_col: str = "company"
    target_col: str = "main_label"


DEFAULT_FOLD_CONFIG = FoldConfig()


def build_fold_assignments(
    df: pd.DataFrame,
    config: FoldConfig | None = None,
) -> np.ndarray:
    """Assign each row to a deterministic company-grouped, country-preserving fold.

    This intentionally mirrors the original benchmark script: companies are shuffled
    independently inside each country with ``np.random.RandomState`` and then assigned
    round-robin to folds. Given the same parquet row order and seed, the output is
    stable.
    """
    config = config or DEFAULT_FOLD_CONFIG
    missing = {config.country_col, config.group_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fold column(s): {sorted(missing)}")

    rng = np.random.RandomState(config.random_state)
    group_to_fold: dict[tuple[object, object], int] = {}

    for country, sub in df.groupby(config.country_col):
        companies = sub[config.group_col].dropna().unique().copy()
        rng.shuffle(companies)
        for idx, company in enumerate(companies):
            group_to_fold[(country, company)] = idx % config.n_splits

    folds = np.empty(len(df), dtype=np.int64)
    fold_columns = df[[config.country_col, config.group_col]]
    for pos, row in enumerate(fold_columns.itertuples(index=False)):
        key = (row[0], row[1])
        try:
            folds[pos] = group_to_fold[key]
        except KeyError as exc:
            raise ValueError(f"Could not assign fold for row {pos}: {key}") from exc
    return folds


def split_indices_for_fold(
    fold_assignments: np.ndarray,
    fold: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fold < 0 or fold >= n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}], got {fold}")

    val_fold = fold
    test_fold = (fold + 1) % n_splits
    train_mask = (fold_assignments != val_fold) & (fold_assignments != test_fold)
    val_mask = fold_assignments == val_fold
    test_mask = fold_assignments == test_fold

    return (
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(test_mask)[0],
    )


def write_split_files(
    df: pd.DataFrame,
    output_dir: str | Path,
    config: FoldConfig | None = None,
) -> np.ndarray:
    config = config or DEFAULT_FOLD_CONFIG
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fold_assignments = build_fold_assignments(df, config)
    np.savetxt(output_path / "fold_assignments.txt", fold_assignments, fmt="%d")

    for fold in range(config.n_splits):
        train_idx, val_idx, test_idx = split_indices_for_fold(
            fold_assignments, fold, config.n_splits
        )
        np.savetxt(output_path / f"split_{fold}_train_idx.txt", train_idx, fmt="%d")
        np.savetxt(output_path / f"split_{fold}_val_idx.txt", val_idx, fmt="%d")
        np.savetxt(output_path / f"split_{fold}_test_idx.txt", test_idx, fmt="%d")

    metadata = build_fold_metadata(df, fold_assignments, config)
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return fold_assignments


def build_fold_metadata(
    df: pd.DataFrame,
    fold_assignments: np.ndarray,
    config: FoldConfig | None = None,
) -> dict[str, Any]:
    config = config or DEFAULT_FOLD_CONFIG
    if len(df) != len(fold_assignments):
        raise ValueError(
            "fold_assignments length must match dataframe length: "
            f"{len(fold_assignments)} != {len(df)}"
        )

    metadata: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_splits": config.n_splits,
        "random_state": config.random_state,
        "country_col": config.country_col,
        "group_col": config.group_col,
        "target_col": config.target_col,
        "fold_sizes": _fold_sizes(fold_assignments, config.n_splits),
        "splits": {},
    }

    if config.target_col in df.columns:
        y = df[config.target_col].astype(int).to_numpy()
        metadata["class_counts"] = _class_counts(y)
    else:
        y = None
        metadata["class_counts"] = None

    for fold in range(config.n_splits):
        train_idx, val_idx, test_idx = split_indices_for_fold(
            fold_assignments, fold, config.n_splits
        )
        metadata["splits"][str(fold)] = {
            "train": _split_summary(train_idx, y),
            "val": _split_summary(val_idx, y),
            "test": _split_summary(test_idx, y),
        }

    return metadata


def _fold_sizes(fold_assignments: np.ndarray, n_splits: int) -> dict[str, int]:
    return {
        str(fold): int((fold_assignments == fold).sum()) for fold in range(n_splits)
    }


def _split_summary(indices: np.ndarray, y: np.ndarray | None) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_rows": int(len(indices))}
    if y is not None:
        summary["class_counts"] = _class_counts(y[indices])
    return summary


def _class_counts(y: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    result = {"0": 0, "1": 0}
    result.update(
        {
            str(int(value)): int(count)
            for value, count in zip(values, counts, strict=True)
        }
    )
    return result
