import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4finbench.data.schema import COUNTRY_COL, FIRM_COL

RESULT_KEY = ["model", "horizon", "test_year"]


@dataclass(frozen=True)
class RunResult:
    model: str
    horizon: int
    test_year: int
    validation_f1: float
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    test_idx: np.ndarray | None = field(default=None, repr=False)
    test_labels: np.ndarray | None = field(default=None, repr=False)
    test_scores: np.ndarray | None = field(default=None, repr=False)
    unseen_firm: np.ndarray | None = field(default=None, repr=False)

    def to_row(self) -> dict[str, Any]:
        details = {
            key: json.dumps(value, sort_keys=True, default=str)
            if isinstance(value, dict | list)
            else value
            for key, value in self.details.items()
        }
        return {
            "model": self.model,
            "horizon": self.horizon,
            "test_year": self.test_year,
            "validation_f1": self.validation_f1,
            **details,
            **self.metrics,
        }


def upsert_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    key: list[str] | None = None,
) -> pd.DataFrame:
    key = key or RESULT_KEY
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    merged = new.drop_duplicates(subset=key, keep="last")
    merged.to_csv(path, index=False)
    return merged


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "host": platform.node(),
        "machine": platform.machine(),
        "n_cpus": os.cpu_count(),
    }
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
    return info


def write_test_scores(path: str | Path, result: RunResult, df: pd.DataFrame) -> Path:
    if result.test_idx is None or result.test_scores is None:
        raise ValueError("RunResult has no test scores to write.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": result.model,
            "horizon": result.horizon,
            "test_year": result.test_year,
            "row_idx": np.asarray(result.test_idx, dtype=np.int64),
            FIRM_COL: df[FIRM_COL].to_numpy()[result.test_idx],
            COUNTRY_COL: df[COUNTRY_COL].to_numpy()[result.test_idx],
            "y_true": result.test_labels,
            "score": result.test_scores,
            "y_pred": result.test_scores >= result.metrics["threshold"],
            "unseen_firm": result.unseen_firm,
        }
    ).to_parquet(path, index=False)
    return path
