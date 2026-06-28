import numpy as np
import pandas as pd
import pytest

from v4finbench.data.horizon_joining import join_horizon_split_frames


def test_join_horizon_split_frames_preserves_splits_and_adds_horizon() -> None:
    data_by_horizon = {
        0: pd.DataFrame({"feature": [0, 1, 2], "main_label": [0, 1, 0]}),
        1: pd.DataFrame({"feature": [10, 11, 12], "main_label": [1, 0, 1]}),
    }
    splits_by_horizon = {
        0: (np.array([0]), np.array([1]), np.array([2])),
        1: (np.array([2]), np.array([0]), np.array([1])),
    }

    joined = join_horizon_split_frames(data_by_horizon, splits_by_horizon)

    assert joined.train["feature"].tolist() == [0, 12]
    assert joined.val["feature"].tolist() == [1, 10]
    assert joined.test["feature"].tolist() == [2, 11]
    assert joined.train["prediction_horizon"].tolist() == [0, 1]
    assert joined.val["prediction_horizon"].tolist() == [0, 1]
    assert joined.test["prediction_horizon"].tolist() == [0, 1]


def test_join_horizon_split_frames_rejects_existing_horizon_column() -> None:
    data_by_horizon = {
        0: pd.DataFrame(
            {
                "feature": [0],
                "prediction_horizon": [99],
                "main_label": [0],
            }
        ),
    }
    splits_by_horizon = {
        0: (np.array([0]), np.array([0]), np.array([0])),
    }

    with pytest.raises(ValueError, match="column already exists"):
        join_horizon_split_frames(data_by_horizon, splits_by_horizon)
