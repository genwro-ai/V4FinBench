import numpy as np
import pandas as pd
import pytest

from v4finbench.data.splits import SplitConfig, TemporalSplit, get_split


@pytest.fixture
def small_splits() -> SplitConfig:
    return SplitConfig(
        first_test_year=2012,
        last_test_year=2013,
        min_train_positives=1,
        min_val_positives=1,
        min_test_positives=1,
    )


@pytest.fixture
def labeled_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2008, 2014):
        for firm in range(6):
            label = int(firm >= 4)
            new_firm = year >= 2012 and firm % 2
            rows.append(
                {
                    "emis_id": f"new{firm}" if new_firm else f"f{firm}",
                    "year": year,
                    "feature": label * 2.0 + rng.normal(scale=0.3),
                    "has_multiple_industries": bool(firm % 2),
                    "main_label": label,
                    "label_observed_year": year + 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def split_2012(labeled_frame: pd.DataFrame, small_splits: SplitConfig) -> TemporalSplit:
    return get_split(labeled_frame, horizon=1, test_year=2012, config=small_splits)
