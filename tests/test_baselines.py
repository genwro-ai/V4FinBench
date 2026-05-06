import numpy as np
import pandas as pd

from v4finbench.models.baselines import normalize_param_grid, train_evaluate_baseline


def test_normalize_param_grid_wraps_scalar_values() -> None:
    grid = normalize_param_grid({"C": [1.0, 2.0], "penalty": "l2"})

    assert grid == {"C": [1.0, 2.0], "penalty": ["l2"]}


def test_train_evaluate_baseline_smoke_logistic_regression() -> None:
    df = pd.DataFrame(
        {
            "company": [f"c{i}" for i in range(12)],
            "feature": [0.0, 0.1, 0.2, 0.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3],
            "has_multiple_industries": [
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "main_label": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    result = train_evaluate_baseline(
        df=df,
        train_idx=np.array([0, 1, 4, 5, 8, 9]),
        val_idx=np.array([2, 6, 10]),
        test_idx=np.array([3, 7, 11]),
        model_name="logistic_regression",
        param_grid={"C": [1.0], "penalty": ["l2"]},
        horizon=0,
        fold=0,
        max_candidates=None,
        output_dir=None,
    )

    assert result.model_name == "logistic_regression"
    assert result.horizon == 0
    assert result.fold == 0
    assert "f1" in result.metrics
