from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from v4finbench.data.preprocessing import (
    preprocess_train_val_test,
    split_features_target,
)
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.thresholds import find_best_f1_threshold

BASELINE_MODEL_NAMES = [
    "logistic_regression",
    "mlp",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]


@dataclass(frozen=True)
class BaselineRunResult:
    model_name: str
    horizon: int
    fold: int
    best_params: dict[str, Any]
    validation_f1: float
    metrics: dict[str, float]


def train_evaluate_baseline(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    param_grid: dict[str, list[Any]],
    horizon: int,
    fold: int,
    target_col: str = "main_label",
    random_state: int = 42,
    max_candidates: int | None = None,
    output_dir: str | Path | None = None,
    progress: bool = False,
) -> BaselineRunResult:
    prepared = split_features_target(df, target_col=target_col)
    X_train_raw = prepared.X.iloc[train_idx]
    X_val_raw = prepared.X.iloc[val_idx]
    X_test_raw = prepared.X.iloc[test_idx]
    y_train = prepared.y.iloc[train_idx].to_numpy()
    y_val = prepared.y.iloc[val_idx].to_numpy()
    y_test = prepared.y.iloc[test_idx].to_numpy()

    X_train, X_val, X_test, preprocessor = preprocess_train_val_test(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )

    best_model = None
    best_params: dict[str, Any] | None = None
    best_threshold = 0.5
    best_validation_f1 = -np.inf

    candidates = iter_param_candidates(param_grid, max_candidates, random_state)
    if progress:
        candidates = tqdm(
            candidates,
            desc=f"{model_name} h={horizon} fold={fold} candidates",
            leave=False,
        )

    for params in candidates:
        model = make_baseline_model(model_name, params, random_state)
        model.fit(X_train, y_train)
        val_score = predict_positive_score(model, X_val)
        threshold = find_best_f1_threshold(y_val, val_score)
        val_metrics = binary_classification_metrics(y_val, val_score, threshold)
        validation_f1 = val_metrics["f1"]

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_params = params
            best_model = model
            best_threshold = threshold

    if best_model is None or best_params is None:
        raise RuntimeError(f"No successful candidate for model {model_name}")

    test_score = predict_positive_score(best_model, X_test)
    metrics = binary_classification_metrics(y_test, test_score, best_threshold)

    if output_dir is not None:
        save_baseline_artifacts(
            output_dir=Path(output_dir),
            model_name=model_name,
            model=best_model,
            preprocessor=preprocessor,
            threshold=best_threshold,
            params=best_params,
        )

    return BaselineRunResult(
        model_name=model_name,
        horizon=horizon,
        fold=fold,
        best_params=best_params,
        validation_f1=float(best_validation_f1),
        metrics=metrics,
    )


def iter_param_candidates(
    param_grid: dict[str, list[Any]],
    max_candidates: int | None,
    random_state: int,
) -> Iterable[dict[str, Any]]:
    param_grid = normalize_param_grid(param_grid)
    if max_candidates is None:
        return list(ParameterGrid(param_grid))
    return list(
        ParameterSampler(
            param_grid,
            n_iter=max_candidates,
            random_state=random_state,
        )
    )


def normalize_param_grid(param_grid: dict[str, Any]) -> dict[str, list[Any]]:
    normalized = {}
    for key, value in param_grid.items():
        if isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = [value]
    return normalized


def make_baseline_model(
    model_name: str,
    params: dict[str, Any],
    random_state: int,
):
    params = dict(params)
    params.setdefault("random_state", random_state)

    if model_name == "logistic_regression":
        params.setdefault("max_iter", 1000)
        params.setdefault("solver", "lbfgs")
        return LogisticRegression(**params)
    if model_name == "mlp":
        params.setdefault("max_iter", 300)
        return MLPClassifier(**params)
    if model_name == "random_forest":
        params.setdefault("n_jobs", -1)
        return RandomForestClassifier(**params)
    if model_name == "xgboost":
        params.setdefault("eval_metric", "logloss")
        params.setdefault("n_jobs", -1)
        return XGBClassifier(**params)
    if model_name == "lightgbm":
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", -1)
        return LGBMClassifier(**params)
    if model_name == "catboost":
        params.setdefault("verbose", False)
        return CatBoostClassifier(**params)

    raise ValueError(f"Unknown baseline model: {model_name}")


def predict_positive_score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def save_baseline_artifacts(
    output_dir: Path,
    model_name: str,
    model,
    preprocessor,
    threshold: float,
    params: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"{model_name}.joblib")
    joblib.dump(preprocessor, output_dir / f"{model_name}_preprocessor.joblib")
    (output_dir / f"{model_name}_threshold.txt").write_text(
        str(threshold),
        encoding="utf-8",
    )
    (output_dir / f"{model_name}_params.json").write_text(
        pd.Series(params).to_json(indent=2),
        encoding="utf-8",
    )
