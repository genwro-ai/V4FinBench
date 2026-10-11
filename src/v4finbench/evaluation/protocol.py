import numpy as np
import pandas as pd

from v4finbench.data.schema import FIRM_COL
from v4finbench.evaluation.metrics import binary_classification_metrics


def unseen_firm_mask(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    firm_col: str = FIRM_COL,
) -> np.ndarray:
    firms = df[firm_col].to_numpy()
    return ~np.isin(firms[test_idx], firms[train_idx])


def evaluate_test_scores(
    y_test: np.ndarray,
    test_score: np.ndarray,
    threshold: float,
    unseen: np.ndarray,
) -> dict[str, float]:
    metrics = binary_classification_metrics(y_test, test_score, threshold)
    metrics["unseen_frac"] = float(unseen.mean()) if len(unseen) else np.nan
    unseen_metrics = (
        binary_classification_metrics(y_test[unseen], test_score[unseen], threshold)
        if unseen.any()
        else {}
    )
    metrics["unseen_roc_auc"] = unseen_metrics.get("roc_auc", np.nan)
    metrics["unseen_pr_auc"] = unseen_metrics.get("pr_auc", np.nan)
    return metrics
