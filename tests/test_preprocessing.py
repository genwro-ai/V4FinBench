import numpy as np
import pandas as pd
import pytest

from v4finbench.data.preprocessing import feature_columns, prepare_split


def test_feature_columns_exclude_ids_time_and_leaky_columns() -> None:
    df = pd.DataFrame(
        columns=[
            "emis_id",
            "company",
            "year",
            "label_observed_year",
            "operational_status",
            "feature",
            "main_label",
        ]
    )

    assert feature_columns(df) == ["feature"]


def test_prepare_split_fits_imputation_and_scaling_on_training_rows() -> None:
    df = pd.DataFrame(
        {"x": [0.0, None, 2.0, None, 100.0], "main_label": [0, 1, 0, 1, 1]}
    )
    data = prepare_split(df, np.array([0, 1, 2]), np.array([3]), np.array([4]))

    assert data.X_train["x"].tolist() == pytest.approx([-1.5**0.5, 0.0, 1.5**0.5])
    assert data.X_val["x"].tolist() == [0.0]
    assert data.y_test.tolist() == [1]
