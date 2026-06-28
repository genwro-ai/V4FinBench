from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JoinedHorizonSplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


SplitIndices = tuple[np.ndarray, np.ndarray, np.ndarray]


def _select_with_horizon(
    df: pd.DataFrame,
    indices: np.ndarray,
    horizon: int,
    horizon_col: str,
) -> pd.DataFrame:
    if horizon_col in df.columns:
        raise ValueError(
            f"Cannot add horizon feature {horizon_col!r}; column already exists."
        )
    selected = df.iloc[np.asarray(indices, dtype=int)].copy()
    selected[horizon_col] = horizon
    return selected


def join_horizon_split_frames(
    data_by_horizon: Mapping[int, pd.DataFrame],
    splits_by_horizon: Mapping[int, SplitIndices],
    horizon_col: str = "prediction_horizon",
) -> JoinedHorizonSplitFrames:
    """Join per-horizon train/val/test splits with horizon as a feature column."""
    if not data_by_horizon:
        raise ValueError("At least one horizon dataframe is required.")

    missing_splits = set(data_by_horizon) - set(splits_by_horizon)
    if missing_splits:
        missing = sorted(missing_splits)
        raise ValueError(f"Missing split indices for horizons: {missing}")

    train_parts = []
    val_parts = []
    test_parts = []
    for horizon in sorted(data_by_horizon):
        df = data_by_horizon[horizon]
        train_idx, val_idx, test_idx = splits_by_horizon[horizon]
        train_parts.append(_select_with_horizon(df, train_idx, horizon, horizon_col))
        val_parts.append(_select_with_horizon(df, val_idx, horizon, horizon_col))
        test_parts.append(_select_with_horizon(df, test_idx, horizon, horizon_col))

    return JoinedHorizonSplitFrames(
        train=pd.concat(train_parts, ignore_index=True),
        val=pd.concat(val_parts, ignore_index=True),
        test=pd.concat(test_parts, ignore_index=True),
    )
