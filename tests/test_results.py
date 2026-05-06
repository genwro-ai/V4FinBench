import pandas as pd
import pytest

from v4finbench.evaluation.results import aggregate_metrics


def test_aggregate_metrics_computes_fold_summary() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["xgboost", "xgboost", "mlp"],
            "horizon": [0, 0, 0],
            "fold": [0, 1, 0],
            "f1": [0.2, 0.4, 0.1],
            "roc_auc": [0.8, 0.9, 0.7],
        }
    )

    summary = aggregate_metrics(metrics)
    xgb = summary[summary["model"] == "xgboost"].iloc[0]

    assert xgb["f1_mean"] == pytest.approx(0.3)
    assert xgb["roc_auc_mean"] == pytest.approx(0.85)
    assert xgb["n_folds"] == 2
