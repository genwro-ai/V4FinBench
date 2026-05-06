import numpy as np

from v4finbench.sampling.prototypes import closest_points_to_centers, create_prototypes
from v4finbench.sampling.strategies import SamplingConfig, apply_sampling


def test_closest_points_to_centers_returns_real_rows() -> None:
    X = np.array([[0.0], [2.0], [10.0]])
    centers = np.array([[1.8], [9.0]])

    closest = closest_points_to_centers(X, centers)

    np.testing.assert_array_equal(closest, np.array([[2.0], [10.0]]))


def test_create_prototypes_is_deterministic() -> None:
    X = np.arange(20, dtype=float).reshape(-1, 1)

    first = create_prototypes(X, n_prototypes=4, random_state=42)
    second = create_prototypes(X, n_prototypes=4, random_state=42)

    np.testing.assert_array_equal(first, second)


def test_random_undersampling_reaches_requested_ratio() -> None:
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 1, 1, *([0] * 20)])

    _, y_sampled = apply_sampling(
        X,
        y,
        SamplingConfig(strategy="undersample", minority_to_majority_ratio=0.5),
    )

    assert int((y_sampled == 1).sum()) == 4
    assert int((y_sampled == 0).sum()) == 8


def test_prototype_undersampling_keeps_minority_and_majority_prototypes() -> None:
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 1, 1, *([0] * 20)])

    X_sampled, y_sampled = apply_sampling(
        X,
        y,
        SamplingConfig(
            strategy="prototype_undersampling",
            minority_to_majority_ratio=0.5,
            use_mini_batch=False,
        ),
    )

    assert X_sampled.shape == (12, 1)
    assert int((y_sampled == 1).sum()) == 4
    assert int((y_sampled == 0).sum()) == 8


def test_no_sampling_returns_inputs_unchanged() -> None:
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])

    X_sampled, y_sampled = apply_sampling(X, y, SamplingConfig(strategy="none"))

    np.testing.assert_array_equal(X_sampled, X)
    np.testing.assert_array_equal(y_sampled, y)
