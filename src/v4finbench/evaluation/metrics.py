import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

FPR_BUDGETS = (0.01, 0.001)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score))
        if both_classes
        else np.nan,
        "threshold": float(threshold),
    }
    for budget in FPR_BUDGETS:
        metrics[f"recall_at_fpr_{budget:g}"] = (
            recall_at_fpr(y_true, y_score, budget) if both_classes else np.nan
        )
    return metrics


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(tpr[fpr <= max_fpr].max())
