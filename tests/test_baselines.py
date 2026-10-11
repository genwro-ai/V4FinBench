import sys

import optuna
import pytest

from v4finbench.models.baselines import run_baseline, suggest

CONFIG = {
    "n_trials": 2,
    "early_stopping_rounds": 5,
    "models": {
        "logistic_regression": {
            "search": {"C": {"low": 0.01, "high": 1.0, "log": True}}
        },
        "xgboost": {"search": {"max_depth": {"low": 2, "high": 3}}},
        "realmlp": {"params": {"n_epochs": 2, "device": "cpu"}},
    },
}


def test_suggest_infers_int_and_float_ranges() -> None:
    trial = optuna.create_study().ask()

    assert isinstance(suggest(trial, "depth", {"low": 2, "high": 8}), int)
    assert isinstance(suggest(trial, "lr", {"low": 0.01, "high": 0.1}), float)
    assert suggest(trial, "sizes", {"choices": [[256, 256], [512]]}) in (
        [256, 256],
        [512],
    )


@pytest.mark.parametrize("model_name", ["logistic_regression", "xgboost"])
def test_run_baseline(labeled_frame, split_2012, model_name) -> None:
    result = run_baseline(labeled_frame, split_2012, model_name, CONFIG)

    assert (result.model, result.horizon, result.test_year) == (model_name, 1, 2012)
    assert result.details["n_trials"] == 2
    assert result.metrics["roc_auc"] > 0.9
    assert result.metrics["unseen_frac"] == 0.5
    assert len(result.test_scores) == len(split_2012.test_idx)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="torch and xgboost/lightgbm load clashing OpenMP runtimes on macOS",
)
def test_run_baseline_realmlp(labeled_frame, split_2012) -> None:
    pytest.importorskip("pytabkit")
    result = run_baseline(labeled_frame, split_2012, "realmlp", CONFIG)

    assert result.details["n_trials"] == 1
    assert len(result.test_scores) == len(split_2012.test_idx)
