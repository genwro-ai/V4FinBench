import numpy as np

from v4finbench.evaluation.metrics import binary_classification_metrics, recall_at_fpr
from v4finbench.evaluation.thresholds import find_best_f1_threshold


def test_find_best_f1_threshold() -> None:
    threshold = find_best_f1_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )

    assert threshold == 0.8


def test_binary_classification_metrics_uses_thresholded_scores() -> None:
    metrics = binary_classification_metrics(
        np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.8, 0.7]), threshold=0.5
    )

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_recall_at_fpr() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])

    assert recall_at_fpr(y_true, y_score, 0.0) == 0.5
    assert recall_at_fpr(y_true, y_score, 0.25) == 1.0


def test_metrics_are_nan_without_positives() -> None:
    metrics = binary_classification_metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.5)

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["recall_at_fpr_0.01"])
