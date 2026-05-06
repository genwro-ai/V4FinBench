from dataclasses import dataclass
from typing import Literal

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer

from v4finbench.sampling.prototypes import create_prototypes

SamplingStrategy = Literal[
    "none",
    "oversample",
    "undersample",
    "prototype_undersampling",
]


@dataclass(frozen=True)
class SamplingConfig:
    strategy: SamplingStrategy = "none"
    random_state: int = 42
    minority_to_majority_ratio: float = 0.3
    use_mini_batch: bool = True


def apply_sampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: SamplingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config = config or SamplingConfig()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).astype(int)

    if config.strategy == "none":
        return X_train, y_train
    _validate_binary_labels(y_train)

    X_train = _impute_if_needed(X_train)

    if config.strategy == "oversample":
        sampler = RandomOverSampler(
            random_state=config.random_state,
            sampling_strategy=config.minority_to_majority_ratio,
        )
        return _sample_with_imblearn(sampler, X_train, y_train, config.random_state)

    if config.strategy == "undersample":
        sampler = RandomUnderSampler(
            random_state=config.random_state,
            sampling_strategy=config.minority_to_majority_ratio,
        )
        return _sample_with_imblearn(sampler, X_train, y_train, config.random_state)

    if config.strategy == "prototype_undersampling":
        return prototype_undersample(X_train, y_train, config)

    raise ValueError(f"Unknown sampling strategy: {config.strategy}")


def prototype_undersample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: SamplingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    class_counts = np.bincount(y_train)
    minority_class = int(np.argmin(class_counts))
    majority_class = 1 - minority_class

    minority_indices = np.where(y_train == minority_class)[0]
    majority_indices = np.where(y_train == majority_class)[0]
    minority_samples = X_train[minority_indices]
    majority_samples = X_train[majority_indices]

    n_majority = int(len(minority_samples) / config.minority_to_majority_ratio)
    majority_prototypes = create_prototypes(
        majority_samples,
        n_prototypes=n_majority,
        random_state=config.random_state,
        use_mini_batch=config.use_mini_batch,
    )

    X_sampled = np.vstack([minority_samples, majority_prototypes])
    y_sampled = np.hstack(
        [
            np.full(len(minority_samples), minority_class, dtype=int),
            np.full(len(majority_prototypes), majority_class, dtype=int),
        ]
    )
    return _shuffle(X_sampled, y_sampled, config.random_state)


def _sample_with_imblearn(
    sampler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_sampled, y_sampled = sampler.fit_resample(X_train, y_train)
    return _shuffle(X_sampled, y_sampled, random_state)


def _shuffle(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.RandomState(random_state).permutation(len(y))
    return X[order], y[order]


def _impute_if_needed(X: np.ndarray) -> np.ndarray:
    if np.isnan(np.asarray(X, dtype=np.float64)).any():
        return SimpleImputer(strategy="median").fit_transform(X)
    return X


def _validate_binary_labels(y: np.ndarray) -> None:
    labels = np.unique(y)
    if len(labels) != 2 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Sampling requires binary labels encoded as 0 and 1.")
