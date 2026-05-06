import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans


def closest_points_to_centers(X: NDArray, centers: NDArray) -> NDArray:
    X_array = np.asarray(X, dtype=np.float64)
    centers_array = np.asarray(centers, dtype=np.float64)
    distances = ((X_array[:, None, :] - centers_array[None, :, :]) ** 2).sum(axis=2)
    closest_indices = np.argmin(distances, axis=0)
    return X_array[closest_indices]


def create_prototypes(
    X: NDArray,
    n_prototypes: int,
    random_state: int = 42,
    use_mini_batch: bool = True,
) -> NDArray:
    X_array = np.asarray(X, dtype=np.float64)
    if n_prototypes <= 0:
        return np.empty((0, X_array.shape[1]), dtype=np.float64)
    if len(X_array) <= n_prototypes:
        return X_array.copy()

    n_clusters = min(n_prototypes, len(X_array))
    kmeans_cls = MiniBatchKMeans if use_mini_batch else KMeans
    kwargs = {
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": 1,
        "init": "random",
    }
    if use_mini_batch:
        kwargs["batch_size"] = 1024

    kmeans = kmeans_cls(**kwargs)
    kmeans.fit(X_array)
    return closest_points_to_centers(X_array, kmeans.cluster_centers_)
