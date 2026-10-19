import numpy as np

from v4finbench.models.tabpfn.sampling import (
    SamplingConfig,
    apply_sampling,
    kmeans_prototypes,
)


def _clusters(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return np.vstack([c + rng.normal(scale=0.5, size=(50, 2)) for c in centers])


def test_kmeans_prototypes_pick_one_real_row_per_cluster(torch_module) -> None:
    X = _clusters()
    index = kmeans_prototypes(X, 3, device="cpu")

    assert sorted(index // 50) == [0, 1, 2]


def test_kmeans_prototypes_are_deterministic_on_cpu(torch_module) -> None:
    X = _clusters()

    np.testing.assert_array_equal(
        kmeans_prototypes(X, 10, device="cpu"), kmeans_prototypes(X, 10, device="cpu")
    )


def test_prototype_undersampling_keeps_all_positives(torch_module) -> None:
    X = _clusters()
    y = np.zeros(len(X), dtype=int)
    y[:6] = 1
    config = SamplingConfig(
        strategy="prototype_undersampling", minority_to_majority_ratio=0.5, device="cpu"
    )

    X_sampled, y_sampled = apply_sampling(X, y, config)

    assert int(y_sampled.sum()) == 6
    assert int((y_sampled == 0).sum()) == 12
    assert set(map(tuple, X_sampled)) <= set(map(tuple, X))


def test_prototype_selection_is_cached(torch_module, tmp_path) -> None:
    X = _clusters()
    y = np.zeros(len(X), dtype=int)
    y[:6] = 1
    config = SamplingConfig(
        strategy="prototype_undersampling", device="cpu", cache_dir=str(tmp_path)
    )

    first = apply_sampling(X, y, config)
    assert len(list(tmp_path.glob("*.npy"))) == 1
    second = apply_sampling(X, y, config)
    np.testing.assert_array_equal(first[0], second[0])

    apply_sampling(X + 1.0, y, config)
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_random_undersampling_reaches_requested_ratio() -> None:
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 1, 1, *([0] * 20)])

    _, y_sampled = apply_sampling(
        X, y, SamplingConfig(strategy="undersample", minority_to_majority_ratio=0.5)
    )

    assert int((y_sampled == 1).sum()) == 4
    assert int((y_sampled == 0).sum()) == 8


def test_no_sampling_returns_inputs_unchanged() -> None:
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])

    X_sampled, y_sampled = apply_sampling(X, y, SamplingConfig(strategy="none"))

    np.testing.assert_array_equal(X_sampled, X)
    np.testing.assert_array_equal(y_sampled, y)
