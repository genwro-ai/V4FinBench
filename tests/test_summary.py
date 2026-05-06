from __future__ import annotations

import pandas as pd

from v4finbench.data.schema import HORIZON_FILES, UNLABELED_FILE
from v4finbench.data.summary import summarize_dataset_dir


def test_summarize_dataset_dir(tmp_path) -> None:
    pd.DataFrame({"company": ["a", "a", "b"]}).to_parquet(
        tmp_path / UNLABELED_FILE,
        index=False,
    )
    pd.DataFrame({"company": ["a", "b", "c"], "main_label": [0, 1, 0]}).to_parquet(
        tmp_path / HORIZON_FILES[0],
        index=False,
    )

    summary = summarize_dataset_dir(tmp_path)
    horizon = summary[summary["horizon"] == 0].iloc[0]

    assert len(summary) == 2
    assert horizon["n_rows"] == 3
    assert horizon["positives"] == 1
    assert horizon["positive_rate"] == 1 / 3

