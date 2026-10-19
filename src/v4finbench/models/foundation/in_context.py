from typing import Any

import numpy as np
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier

from v4finbench.models.foundation.sampling import SamplingConfig, apply_sampling

ESTIMATORS = {"tabpfn": TabPFNClassifier, "tabicl": TabICLClassifier}
PREDICT_CHUNK_ROWS = 50_000


class InContextModel:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.vstack(
            [
                self.model.predict_proba(X[start : start + PREDICT_CHUNK_ROWS])
                for start in range(0, len(X), PREDICT_CHUNK_ROWS)
            ]
        )


def sampling_config(params: dict[str, Any], random_state: int) -> SamplingConfig:
    return SamplingConfig(
        strategy=params.pop("sampling", "none"),
        minority_to_majority_ratio=params.pop("ratio", 0.3),
        random_state=random_state,
        device=params.pop("sampling_device", "auto"),
        cache_dir=params.pop("sampling_cache_dir", None),
    )


def fit_in_context(
    estimator: str, params: dict[str, Any], X, y, random_state: int
) -> InContextModel:
    params = dict(params)
    sampling = sampling_config(params, random_state)
    X_context, y_context = apply_sampling(np.asarray(X, dtype=np.float32), y, sampling)
    if estimator == "tabpfn":
        params.setdefault("ignore_pretraining_limits", True)
    model = ESTIMATORS[estimator](random_state=random_state, **params)
    return InContextModel(model.fit(X_context, y_context))
