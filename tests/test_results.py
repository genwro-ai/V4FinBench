import json

import pandas as pd

from v4finbench.evaluation.results import RunResult, upsert_rows


def test_upsert_rows_replaces_rows_with_the_same_key(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    row = RunResult("xgboost", 1, 2015, 0.3, {"roc_auc": 0.8}, {"p": {"a": 1}})
    upsert_rows(path, [row.to_row()])
    upsert_rows(path, [{**row.to_row(), "roc_auc": 0.9}])

    metrics = pd.read_csv(path)
    assert metrics["roc_auc"].tolist() == [0.9]
    assert json.loads(metrics["p"].iloc[0]) == {"a": 1}
