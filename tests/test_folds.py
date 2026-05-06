import numpy as np
import pandas as pd

from v4finbench.data.folds import (
    FoldConfig,
    build_fold_assignments,
    build_fold_metadata,
    split_indices_for_fold,
)


def test_fold_assignments_are_deterministic() -> None:
    df = pd.DataFrame(
        {
            "country": ["PL", "PL", "PL", "CZ", "CZ", "CZ", "CZ"],
            "company": ["a", "a", "b", "c", "d", "d", "e"],
            "year": [2019, 2020, 2020, 2018, 2018, 2019, 2020],
        }
    )

    config = FoldConfig(n_splits=3, random_state=42)
    first = build_fold_assignments(df, config)
    second = build_fold_assignments(df, config)

    np.testing.assert_array_equal(first, second)


def test_company_rows_stay_in_same_fold_within_country() -> None:
    df = pd.DataFrame(
        {
            "country": ["PL", "PL", "PL", "CZ", "CZ", "CZ"],
            "company": ["a", "a", "b", "a", "a", "c"],
        }
    )

    folds = build_fold_assignments(df, FoldConfig(n_splits=2, random_state=42))
    assigned = df.assign(fold=folds)

    sizes = assigned.groupby(["country", "company"])["fold"].nunique()
    assert sizes.max() == 1


def test_rotating_train_val_test_indices() -> None:
    assignments = np.array([0, 1, 2, 3, 4, 0, 1, 2])

    train_idx, val_idx, test_idx = split_indices_for_fold(assignments, fold=4)

    np.testing.assert_array_equal(val_idx, np.array([4]))
    np.testing.assert_array_equal(test_idx, np.array([0, 5]))
    np.testing.assert_array_equal(train_idx, np.array([1, 2, 3, 6, 7]))


def test_fold_metadata_reports_class_counts() -> None:
    df = pd.DataFrame(
        {
            "country": ["PL"] * 6,
            "company": ["a", "b", "c", "d", "e", "f"],
            "main_label": [0, 1, 0, 1, 0, 0],
        }
    )
    assignments = np.array([0, 0, 1, 1, 2, 2])

    metadata = build_fold_metadata(
        df,
        assignments,
        FoldConfig(n_splits=3, random_state=42),
    )

    assert metadata["n_rows"] == 6
    assert metadata["class_counts"] == {"0": 4, "1": 2}
    assert metadata["fold_sizes"] == {"0": 2, "1": 2, "2": 2}
    assert metadata["splits"]["0"]["val"]["class_counts"] == {"0": 1, "1": 1}
    assert metadata["splits"]["0"]["test"]["class_counts"] == {"0": 1, "1": 1}
    assert metadata["splits"]["0"]["train"]["class_counts"] == {"0": 2, "1": 0}
