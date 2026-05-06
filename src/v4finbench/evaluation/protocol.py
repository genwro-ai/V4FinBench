from dataclasses import dataclass

import numpy as np

from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.thresholds import find_best_f1_threshold


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float
    metrics: dict[str, float]


def evaluate_with_validation_threshold(
    y_val: np.ndarray,
    val_score: np.ndarray,
    y_test: np.ndarray,
    test_score: np.ndarray,
) -> EvaluationResult:
    threshold = find_best_f1_threshold(y_val, val_score)
    metrics = binary_classification_metrics(y_test, test_score, threshold)
    return EvaluationResult(threshold=threshold, metrics=metrics)
