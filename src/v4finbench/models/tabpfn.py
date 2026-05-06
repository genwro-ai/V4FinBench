from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4finbench.data.preprocessing import (
    preprocess_train_val_test,
    split_features_target,
)
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.thresholds import find_best_f1_threshold
from v4finbench.sampling.strategies import SamplingConfig, apply_sampling


@dataclass(frozen=True)
class TabPFNRunConfig:
    sampling_strategy: str = "none"
    n_context_samples: int = 10_000
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    minority_to_majority_ratio: float = 0.3
    random_state: int = 42
    device: str = "auto"
    model_path: str | None = None


@dataclass(frozen=True)
class TabPFNRunResult:
    model_name: str
    horizon: int
    fold: int
    sampling_strategy: str
    n_context_samples: int
    validation_f1: float
    metrics: dict[str, float]


ClassifierFactory = Callable[[TabPFNRunConfig], Any]


def tabpfn_config_from_mapping(values: dict[str, Any]) -> TabPFNRunConfig:
    return TabPFNRunConfig(
        sampling_strategy=values.get("sampling_strategy", "none"),
        n_context_samples=int(values.get("n_context_samples", 10_000)),
        max_train_samples=_optional_int(values.get("max_train_samples")),
        max_eval_samples=_optional_int(values.get("max_eval_samples")),
        minority_to_majority_ratio=float(values.get("minority_to_majority_ratio", 0.3)),
        random_state=int(values.get("random_seed", values.get("random_state", 42))),
        device=values.get("device", "auto"),
        model_path=values.get("model_path"),
    )


def train_evaluate_tabpfn(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon: int,
    fold: int,
    config: TabPFNRunConfig | None = None,
    target_col: str = "main_label",
    classifier_factory: ClassifierFactory | None = None,
) -> TabPFNRunResult:
    config = config or TabPFNRunConfig()
    train_idx = subsample_indices(
        train_idx,
        max_samples=config.max_train_samples,
        random_state=config.random_state,
    )
    val_idx = subsample_indices(
        val_idx,
        max_samples=config.max_eval_samples,
        random_state=config.random_state + 1,
    )
    test_idx = subsample_indices(
        test_idx,
        max_samples=config.max_eval_samples,
        random_state=config.random_state + 2,
    )
    prepared = split_features_target(df, target_col=target_col)
    X_train_raw = prepared.X.iloc[train_idx]
    X_val_raw = prepared.X.iloc[val_idx]
    X_test_raw = prepared.X.iloc[test_idx]
    y_train = prepared.y.iloc[train_idx].to_numpy()
    y_val = prepared.y.iloc[val_idx].to_numpy()
    y_test = prepared.y.iloc[test_idx].to_numpy()

    X_train, X_val, X_test, _ = preprocess_train_val_test(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )
    X_train_np = np.asarray(X_train, dtype=np.float64)
    X_val_np = np.asarray(X_val, dtype=np.float64)
    X_test_np = np.asarray(X_test, dtype=np.float64)

    X_sampled, y_sampled = apply_sampling(
        X_train_np,
        y_train,
        SamplingConfig(
            strategy=config.sampling_strategy,
            random_state=config.random_state,
            minority_to_majority_ratio=config.minority_to_majority_ratio,
        ),
    )
    X_context, y_context = select_context_samples(
        X_sampled,
        y_sampled,
        n_context_samples=config.n_context_samples,
        random_state=config.random_state,
    )

    classifier = (
        classifier_factory(config)
        if classifier_factory is not None
        else make_tabpfn_classifier(config)
    )
    classifier.fit(X_context, y_context)

    val_score = classifier.predict_proba(X_val_np)[:, 1]
    threshold = find_best_f1_threshold(y_val, val_score)
    val_metrics = binary_classification_metrics(y_val, val_score, threshold)

    test_score = classifier.predict_proba(X_test_np)[:, 1]
    metrics = binary_classification_metrics(y_test, test_score, threshold)

    return TabPFNRunResult(
        model_name="tabpfn",
        horizon=horizon,
        fold=fold,
        sampling_strategy=config.sampling_strategy,
        n_context_samples=len(y_context),
        validation_f1=val_metrics["f1"],
        metrics=metrics,
    )


def select_context_samples(
    X: np.ndarray,
    y: np.ndarray,
    n_context_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_context_samples <= 0 or len(y) <= n_context_samples:
        return X, y
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(y), size=n_context_samples, replace=False)
    return X[indices], y[indices]


def subsample_indices(
    indices: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> np.ndarray:
    if max_samples is None:
        return indices
    if max_samples <= 0:
        raise ValueError("max_samples must be positive or None.")
    if len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(random_state)
    selected = rng.choice(len(indices), size=max_samples, replace=False)
    return np.asarray(indices)[selected]


def make_tabpfn_classifier(config: TabPFNRunConfig):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise ImportError(
            "TabPFN is not installed. Run `uv sync --extra tabpfn` first."
        ) from exc

    kwargs: dict[str, Any] = {
        "random_state": config.random_state,
        "ignore_pretraining_limits": True,
    }
    if config.model_path is not None:
        kwargs["model_path"] = config.model_path
    if config.device != "auto":
        kwargs["device"] = config.device
    return TabPFNClassifier(**kwargs)


def save_tabpfn_result(
    output_dir: str | Path,
    result: TabPFNRunResult,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "metrics.csv"
    row = tabpfn_result_to_row(result)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def tabpfn_result_to_row(result: TabPFNRunResult) -> dict[str, Any]:
    row = {
        "model": result.model_name,
        "horizon": result.horizon,
        "fold": result.fold,
        "sampling_strategy": result.sampling_strategy,
        "n_context_samples": result.n_context_samples,
        "validation_f1": result.validation_f1,
    }
    row.update(result.metrics)
    return row


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
