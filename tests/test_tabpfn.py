from __future__ import annotations

import numpy as np
import pandas as pd

from v4finbench.models.tabpfn import (
    TabPFNRunConfig,
    select_context_samples,
    subsample_indices,
    train_evaluate_tabpfn,
)


class FakeTabPFNClassifier:
    def fit(self, X, y):
        self.threshold_ = float(np.median(X[:, 0]))
        return self

    def predict_proba(self, X):
        score = 1 / (1 + np.exp(-(X[:, 0] - self.threshold_)))
        return np.column_stack([1 - score, score])


def test_select_context_samples_is_deterministic() -> None:
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20)

    X_first, y_first = select_context_samples(X, y, 5, random_state=42)
    X_second, y_second = select_context_samples(X, y, 5, random_state=42)

    np.testing.assert_array_equal(X_first, X_second)
    np.testing.assert_array_equal(y_first, y_second)


def test_subsample_indices_is_deterministic() -> None:
    indices = np.arange(20)

    first = subsample_indices(indices, max_samples=5, random_state=42)
    second = subsample_indices(indices, max_samples=5, random_state=42)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 5


def test_train_evaluate_tabpfn_with_fake_classifier() -> None:
    df = pd.DataFrame(
        {
            "company": [f"c{i}" for i in range(12)],
            "feature": [0.0, 0.1, 0.2, 0.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3],
            "main_label": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    result = train_evaluate_tabpfn(
        df=df,
        train_idx=np.array([0, 1, 4, 5, 8, 9]),
        val_idx=np.array([2, 6, 10]),
        test_idx=np.array([3, 7, 11]),
        horizon=0,
        fold=0,
        config=TabPFNRunConfig(
            n_context_samples=4,
            max_train_samples=6,
            max_eval_samples=3,
        ),
        classifier_factory=lambda _: FakeTabPFNClassifier(),
    )

    assert result.model_name == "tabpfn"
    assert result.n_context_samples == 4
    assert "f1" in result.metrics
