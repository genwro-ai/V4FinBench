import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans

PrototypeBackend = str


def closest_points_to_centers(
    X: NDArray,
    centers: NDArray,
    backend: PrototypeBackend = "sklearn",
) -> NDArray:
    X_array = np.asarray(X, dtype=np.float64)
    centers_array = np.asarray(centers, dtype=np.float64)
    if len(centers_array) == 0:
        return np.empty((0, X_array.shape[1]), dtype=np.float64)
    if len(X_array) == 0:
        raise ValueError("Cannot find closest points in an empty sample matrix.")

    normalized = _normalize_backend(backend)
    if normalized == "cuml":
        closest_indices = _closest_points_to_centers_cuml(
            X_array,
            centers_array,
        )
    else:
        closest_indices = _closest_points_to_centers_sklearn(
            X_array,
            centers_array,
        )
    return X_array[closest_indices]


def create_prototypes(
    X: NDArray,
    n_prototypes: int,
    random_state: int = 42,
    use_mini_batch: bool = True,
    backend: PrototypeBackend = "sklearn",
) -> NDArray:
    X_array = np.asarray(X, dtype=np.float64)
    if n_prototypes <= 0:
        return np.empty((0, X_array.shape[1]), dtype=np.float64)
    if len(X_array) <= n_prototypes:
        return X_array.copy()

    n_clusters = min(n_prototypes, len(X_array))
    normalized = _normalize_backend(backend)
    if normalized == "cuml":
        prototypes = _create_prototypes_cuml(
            X_array,
            n_clusters,
            random_state,
        )
    else:
        prototypes = _create_prototypes_sklearn(
            X_array,
            n_clusters,
            random_state,
            use_mini_batch,
        )
    return prototypes


def _create_prototypes_sklearn(
    X_array: NDArray,
    n_clusters: int,
    random_state: int,
    use_mini_batch: bool,
) -> NDArray:
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
    return closest_points_to_centers(
        X_array,
        kmeans.cluster_centers_,
        backend="sklearn",
    )


def _create_prototypes_cuml(
    X_array: NDArray,
    n_clusters: int,
    random_state: int,
) -> NDArray:
    import cupy as cp
    from cuml.cluster import KMeans as CuKMeans

    X_gpu = cp.asarray(X_array, dtype=cp.float32)
    kmeans = CuKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=1,
        init="random",
    )
    kmeans.fit(X_gpu)
    centers_gpu = cp.asarray(kmeans.cluster_centers_, dtype=cp.float32)
    closest_indices = _closest_points_to_centers_cuml_arrays(
        X_gpu,
        centers_gpu,
    )
    return X_array[closest_indices]


def _closest_points_to_centers_sklearn(
    X_array: NDArray,
    centers_array: NDArray,
) -> NDArray:
    from sklearn.metrics import pairwise_distances_argmin_min

    closest_indices, _ = pairwise_distances_argmin_min(centers_array, X_array)
    return np.asarray(closest_indices, dtype=np.int64)


def _closest_points_to_centers_cuml(
    X_array: NDArray,
    centers_array: NDArray,
) -> NDArray:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    X_gpu = cp.asarray(X_array, dtype=cp.float32)
    centers_gpu = cp.asarray(centers_array, dtype=cp.float32)
    return _closest_points_to_centers_cuml_arrays(
        X_gpu,
        centers_gpu,
        NearestNeighbors,
    )


def _closest_points_to_centers_cuml_arrays(
    X_gpu,
    centers_gpu,
    nearest_neighbors_cls=None,
) -> NDArray:
    import cupy as cp

    if nearest_neighbors_cls is None:
        from cuml.neighbors import NearestNeighbors as nearest_neighbors_cls

    nearest = nearest_neighbors_cls(n_neighbors=1)
    nearest.fit(X_gpu)
    _, indices = nearest.kneighbors(centers_gpu)
    return cp.asnumpy(indices).reshape(-1).astype(np.int64)


def _normalize_backend(backend: PrototypeBackend) -> str:
    normalized = str(backend).lower()
    if normalized in {"cuml", "sklearn"}:
        return normalized
    raise ValueError(
        "prototype_backend must be one of 'cuml' or 'sklearn', "
        f"got {backend!r}."
    )
