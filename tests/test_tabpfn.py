import numpy as np
import pytest

from v4finbench.models.tabpfn.vanilla import (
    TabPFNRunConfig,
    run_tabpfn,
    select_context_samples,
    subsample_indices,
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

    X_first, _ = select_context_samples(X, y, 5, random_state=42)
    X_second, _ = select_context_samples(X, y, 5, random_state=42)

    np.testing.assert_array_equal(X_first, X_second)


def test_subsample_indices_rejects_non_positive_max_samples() -> None:
    with pytest.raises(ValueError, match="max_samples must be positive"):
        subsample_indices(np.arange(20), max_samples=0, random_state=42)


def test_run_tabpfn_with_fake_classifier(labeled_frame, split_2012) -> None:
    result = run_tabpfn(
        labeled_frame,
        split_2012,
        TabPFNRunConfig(n_context_samples=4),
        classifier_factory=lambda _: FakeTabPFNClassifier(),
    )

    assert result.model == "tabpfn_none"
    assert result.details["n_context_samples"] == 4
    assert len(result.test_scores) == len(split_2012.test_idx)
