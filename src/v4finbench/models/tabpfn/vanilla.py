import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from v4finbench.data.preprocessing import prepare_split
from v4finbench.data.splits import TemporalSplit
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.protocol import evaluate_test_scores, unseen_firm_mask
from v4finbench.evaluation.results import RunResult
from v4finbench.evaluation.thresholds import find_best_f1_threshold
from v4finbench.models.tabpfn.sampling import SamplingConfig, apply_sampling


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
    sampling_cache_dir: str | None = None


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
        sampling_cache_dir=values.get("sampling_cache_dir"),
    )


def run_tabpfn(
    df: pd.DataFrame,
    split: TemporalSplit,
    config: TabPFNRunConfig | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> RunResult:
    config = config or TabPFNRunConfig()
    train_idx, val_idx, test_idx = subsample_split(
        split, config.max_train_samples, config.max_eval_samples, config.random_state
    )
    data = prepare_split(df, train_idx, val_idx, test_idx)

    X_sampled, y_sampled = apply_sampling(
        np.asarray(data.X_train, dtype=np.float64),
        data.y_train,
        SamplingConfig(
            strategy=config.sampling_strategy,
            random_state=config.random_state,
            minority_to_majority_ratio=config.minority_to_majority_ratio,
            device=config.device,
            cache_dir=config.sampling_cache_dir,
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
    fit_start = time.perf_counter()
    classifier.fit(X_context, y_context)
    val_score = predict_proba_batched(
        classifier, np.asarray(data.X_val, dtype=np.float64)
    )
    threshold = find_best_f1_threshold(data.y_val, val_score)
    validation_f1 = binary_classification_metrics(data.y_val, val_score, threshold)[
        "f1"
    ]
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    test_score = predict_proba_batched(
        classifier, np.asarray(data.X_test, dtype=np.float64)
    )
    predict_seconds = time.perf_counter() - predict_start

    unseen = unseen_firm_mask(df, train_idx, test_idx)
    return RunResult(
        model=f"tabpfn_{config.sampling_strategy}",
        horizon=split.horizon,
        test_year=split.test_year,
        validation_f1=validation_f1,
        metrics=evaluate_test_scores(data.y_test, test_score, threshold, unseen),
        details={
            "sampling_strategy": config.sampling_strategy,
            "n_context_samples": len(y_context),
            "fit_seconds": round(fit_seconds, 2),
            "predict_seconds": round(predict_seconds, 4),
        },
        test_idx=test_idx,
        test_labels=data.y_test,
        test_scores=test_score,
        unseen_firm=unseen,
    )


def subsample_split(
    split: TemporalSplit,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        subsample_indices(split.train_idx, max_train_samples, random_state),
        subsample_indices(split.val_idx, max_eval_samples, random_state + 1),
        subsample_indices(split.test_idx, max_eval_samples, random_state + 2),
    )


def predict_proba_batched(
    classifier,
    X: np.ndarray,
    batch_size: int = 32_768,
) -> np.ndarray:
    # One forward over ~200k query rows needs >20 GiB; chunk to fit an A100.
    scores = [
        classifier.predict_proba(X[start : start + batch_size])[:, 1]
        for start in range(0, len(X), batch_size)
    ]
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)


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


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
