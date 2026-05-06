import numpy as np
import pandas as pd

from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.thresholds import find_best_f1_threshold

LABEL_MAP = {"YES": 1, "NO": 0}


def labels_from_predictions(
    df: pd.DataFrame,
    *,
    label_col: str = "assistant",
    prediction_col: str = "prediction",
) -> tuple[np.ndarray, np.ndarray, int]:
    y_true = _label_series(df, label_col)
    y_pred_raw = df[prediction_col].astype(str).str.strip().str.upper()
    valid = y_true.isin([0, 1]) & y_pred_raw.isin(["YES", "NO"])
    return (
        y_true[valid].astype(int).to_numpy(),
        y_pred_raw[valid].map(LABEL_MAP).astype(int).to_numpy(),
        int((~valid).sum()),
    )


def hard_prediction_metrics(
    df: pd.DataFrame,
    *,
    label_col: str = "assistant",
    prediction_col: str = "prediction",
) -> dict[str, float | int]:
    y_true, y_pred, unknown = labels_from_predictions(
        df,
        label_col=label_col,
        prediction_col=prediction_col,
    )
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "threshold": 0.5,
            "total_rows": int(len(df)),
            "known_predictions": 0,
            "unknown_predictions": int(len(df)),
        }
    metrics = binary_classification_metrics(y_true, y_pred.astype(float), threshold=0.5)
    metrics["total_rows"] = int(len(df))
    metrics["known_predictions"] = int(len(y_true))
    metrics["unknown_predictions"] = unknown
    return metrics


def threshold_probability_predictions(
    df: pd.DataFrame,
    *,
    prob_col: str = "p_yes",
    label_col: str = "assistant",
) -> tuple[pd.DataFrame, dict[str, float]]:
    if prob_col not in df.columns:
        raise ValueError(f"Missing probability column: {prob_col}")
    y_true = _label_series(df, label_col)
    probabilities = pd.to_numeric(df[prob_col], errors="coerce")
    valid = y_true.isin([0, 1]) & probabilities.notna()
    if valid.sum() == 0:
        raise ValueError("No valid labels and probabilities to threshold.")

    y_true_valid = y_true[valid].astype(int).to_numpy()
    p_valid = probabilities[valid].astype(float).to_numpy()
    threshold = find_best_f1_threshold(y_true_valid, p_valid)
    metrics = binary_classification_metrics(y_true_valid, p_valid, threshold)
    out = df.copy()
    out["prediction"] = np.where(probabilities >= threshold, "YES", "NO")
    out["prediction_threshold"] = threshold
    return out, metrics


def _label_series(df: pd.DataFrame, label_col: str) -> pd.Series:
    if label_col in df.columns:
        if label_col == "main_label":
            return pd.to_numeric(df[label_col], errors="coerce")
        return df[label_col].astype(str).str.strip().str.upper().map(LABEL_MAP)
    if "main_label" in df.columns:
        return pd.to_numeric(df["main_label"], errors="coerce")
    raise ValueError(f"Missing label column: {label_col}")
