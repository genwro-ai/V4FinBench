from dataclasses import replace

import pandas as pd
import pytest

from v4finbench.data.splits import SplitConfig, get_split, temporal_splits


def _frame(first_year: int, horizon: int = 1, positives: bool = True) -> pd.DataFrame:
    rows = [
        (year, label, year + horizon)
        for year in range(first_year, 2014)
        for label in (0, int(positives))
    ]
    df = pd.DataFrame(rows, columns=["year", "main_label", "label_observed_year"])
    df["label_observed_year"] = df["label_observed_year"].astype("Int64")
    return df


def _years(df: pd.DataFrame, idx) -> set[int]:
    return set(df["year"].to_numpy()[idx].tolist())


def test_walk_forward_layout(
    labeled_frame: pd.DataFrame, small_splits: SplitConfig
) -> None:
    split = get_split(labeled_frame, 1, 2012, small_splits)

    assert _years(labeled_frame, split.test_idx) == {2012}
    assert _years(labeled_frame, split.val_idx) == {2010, 2011}
    assert _years(labeled_frame, split.train_idx) == {2008, 2009}


def test_training_labels_must_resolve_by_test_year(small_splits: SplitConfig) -> None:
    df = _frame(2004, horizon=2)
    split = get_split(df, 2, 2012, small_splits)

    assert _years(df, split.val_idx) == {2009, 2010}
    assert max(_years(df, split.train_idx)) == 2008


def test_unobserved_labels_never_enter_training(small_splits: SplitConfig) -> None:
    df = _frame(2006)
    df.loc[df["year"] == 2007, "label_observed_year"] = pd.NA
    split = get_split(df, 1, 2012, small_splits)

    assert _years(df, split.train_idx) == {2006, 2008, 2009}


def test_availability_margin_shifts_windows_back(small_splits: SplitConfig) -> None:
    config = replace(small_splits, availability_margin=1)
    df = _frame(2004)

    assert _years(df, get_split(df, 1, 2012, small_splits).val_idx) == {2010, 2011}
    assert _years(df, get_split(df, 1, 2012, config).val_idx) == {2009, 2010}


def test_thin_splits_are_skipped(small_splits: SplitConfig) -> None:
    df = _frame(2004, positives=False)

    assert temporal_splits(df, 1, small_splits) == []
    skipped = temporal_splits(df, 1, small_splits, include_skipped=True)
    assert skipped[0].skip_reason == "few test positives"
    with pytest.raises(ValueError, match="skipped"):
        get_split(df, 1, 2012, small_splits)
