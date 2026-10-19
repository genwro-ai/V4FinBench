import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer

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
    device: str = "auto"
    cache_dir: str | None = None


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
        selected = _cached(prototype_undersample_indices, X_train, y_train, config)
        return X_train[selected], y_train[selected]

    raise ValueError(f"Unknown sampling strategy: {config.strategy}")


def prototype_undersample_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: SamplingConfig,
) -> np.ndarray:
    minority_class = int(np.argmin(np.bincount(y_train)))
    minority = np.flatnonzero(y_train == minority_class)
    majority = np.flatnonzero(y_train != minority_class)
    n_prototypes = int(len(minority) / config.minority_to_majority_ratio)
    prototypes = majority[
        kmeans_prototypes(
            X_train[majority], n_prototypes, config.random_state, config.device
        )
    ]
    selected = np.concatenate([minority, prototypes])
    return np.random.RandomState(config.random_state).permutation(selected)


def kmeans_prototypes(
    X: np.ndarray,
    n_prototypes: int,
    random_state: int = 42,
    device: str = "auto",
    max_iter: int = 100,
    tol: float = 1e-4,
    chunk_size: int = 32_768,
) -> np.ndarray:
    """Indices of the rows nearest to the centres of a k-means fit on ``X``."""
    if n_prototypes >= len(X):
        return np.arange(len(X))
    device = _torch_device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    points = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    squared_norms = (points * points).sum(1)
    generator = torch.Generator().manual_seed(random_state)
    centers = points[torch.randperm(len(points), generator=generator)[:n_prototypes]]

    def nearest(centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        center_norms = (centers * centers).sum(1)
        distances, labels = [], []
        for start in range(0, len(points), chunk_size):
            block = slice(start, start + chunk_size)
            d = (
                squared_norms[block, None]
                - 2 * points[block] @ centers.T
                + center_norms
            )
            min_d, label = d.min(1)
            distances.append(min_d)
            labels.append(label)
        return torch.cat(distances), torch.cat(labels)

    previous = None
    for _ in range(max_iter):
        distances, labels = nearest(centers)
        inertia = distances.clamp(min=0).mean().item()
        if previous is not None and previous - inertia <= tol * previous:
            break
        previous = inertia
        counts = torch.bincount(labels, minlength=n_prototypes)
        sums = torch.zeros_like(centers).index_add_(0, labels, points)
        centers = torch.where(
            counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], centers
        )

    center_norms = (centers * centers).sum(1)
    best = torch.full((n_prototypes,), float("inf"), device=device)
    best_index = torch.zeros(n_prototypes, dtype=torch.long, device=device)
    for start in range(0, len(points), chunk_size):
        block = slice(start, start + chunk_size)
        d = squared_norms[block, None] - 2 * points[block] @ centers.T + center_norms
        min_d, index = d.min(0)
        closer = min_d < best
        best = torch.where(closer, min_d, best)
        best_index = torch.where(closer, index + start, best_index)
    return np.unique(best_index.cpu().numpy())


def _torch_device(device: str):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cached(compute, X: np.ndarray, y: np.ndarray, config: SamplingConfig):
    if config.cache_dir is None:
        return compute(X, y, config)
    key = hashlib.sha256()
    key.update(np.ascontiguousarray(X).tobytes())
    key.update(np.ascontiguousarray(y).tobytes())
    key.update(
        f"{config.strategy}|{config.minority_to_majority_ratio}|"
        f"{config.random_state}".encode()
    )
    path = Path(config.cache_dir) / f"{config.strategy}_{key.hexdigest()[:20]}.npy"
    if path.exists():
        return np.load(path)
    selected = compute(X, y, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, selected)
    return selected


def _sample_with_imblearn(
    sampler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_sampled, y_sampled = sampler.fit_resample(X_train, y_train)
    order = np.random.RandomState(random_state).permutation(len(y_sampled))
    return X_sampled[order], y_sampled[order]


def _impute_if_needed(X: np.ndarray) -> np.ndarray:
    if np.isnan(np.asarray(X, dtype=np.float64)).any():
        return SimpleImputer(strategy="median").fit_transform(X)
    return X


def _validate_binary_labels(y: np.ndarray) -> None:
    labels = np.unique(y)
    if len(labels) != 2 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Sampling requires binary labels encoded as 0 and 1.")
