from __future__ import annotations

import numpy as np

from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.protocol import evaluate_with_validation_threshold
from v4finbench.evaluation.thresholds import find_best_f1_threshold


def test_find_best_f1_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    threshold = find_best_f1_threshold(y_true, y_score)

    assert threshold == 0.8


def test_binary_classification_metrics_uses_thresholded_scores() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.8, 0.7])

    metrics = binary_classification_metrics(y_true, y_score, threshold=0.5)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_evaluate_with_validation_threshold() -> None:
    result = evaluate_with_validation_threshold(
        y_val=np.array([0, 0, 1, 1]),
        val_score=np.array([0.1, 0.2, 0.8, 0.9]),
        y_test=np.array([0, 1]),
        test_score=np.array([0.2, 0.8]),
    )

    assert result.threshold == 0.8
    assert result.metrics["f1"] == 1.0

