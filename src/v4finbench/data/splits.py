from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from v4finbench.data.schema import LABEL_YEAR_COL, TARGET_COL, YEAR_COL


@dataclass(frozen=True)
class SplitConfig:
    first_test_year: int = 2013
    last_test_year: int = 2020
    val_n_years: int = 2
    availability_margin: int = 0
    min_train_positives: int = 30
    min_val_positives: int = 15
    min_test_positives: int = 30


@dataclass(frozen=True)
class TemporalSplit:
    horizon: int
    test_year: int
    train_idx: np.ndarray = field(repr=False)
    val_idx: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)
    skip_reason: str | None = None


def temporal_splits(
    df: pd.DataFrame,
    horizon: int,
    config: SplitConfig | None = None,
    include_skipped: bool = False,
) -> list[TemporalSplit]:
    config = config or SplitConfig()
    years = df[YEAR_COL].to_numpy()
    y = df[TARGET_COL].to_numpy()
    observed = df[LABEL_YEAR_COL].to_numpy(dtype=float, na_value=np.inf)

    splits = []
    for test_year in range(config.first_test_year, config.last_test_year + 1):
        cutoff = test_year - config.availability_margin
        pool = (years < test_year) & (years + horizon <= cutoff) & (observed <= cutoff)
        pool_years = np.unique(years[pool])
        if len(pool_years) <= config.val_n_years:
            continue
        first_val_year = pool_years[-config.val_n_years]
        train = pool & (years < first_val_year)
        val = pool & (years >= first_val_year)
        test = years == test_year

        split = TemporalSplit(
            horizon=horizon,
            test_year=test_year,
            train_idx=np.flatnonzero(train),
            val_idx=np.flatnonzero(val),
            test_idx=np.flatnonzero(test),
            skip_reason=_skip_reason(y[train], y[val], y[test], config),
        )
        if include_skipped or split.skip_reason is None:
            splits.append(split)
    return splits


def get_split(
    df: pd.DataFrame,
    horizon: int,
    test_year: int,
    config: SplitConfig | None = None,
) -> TemporalSplit:
    for split in temporal_splits(df, horizon, config, include_skipped=True):
        if split.test_year == test_year:
            if split.skip_reason is not None:
                raise ValueError(
                    f"Split h={horizon}, test_year={test_year} is skipped: "
                    f"{split.skip_reason}"
                )
            return split
    raise ValueError(f"No split for h={horizon}, test_year={test_year}")


def _skip_reason(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    config: SplitConfig,
) -> str | None:
    if y_test.sum() < config.min_test_positives:
        return "few test positives"
    if y_train.sum() < config.min_train_positives:
        return "few train positives"
    if y_val.sum() < config.min_val_positives:
        return "few val positives"
    return None
