from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from v4finbench.data.schema import FIRM_COL, LABEL_YEAR_COL, TARGET_COL, YEAR_COL

COLUMNS_TO_DROP = [
    TARGET_COL,
    FIRM_COL,
    "company",
    "industry",
    "link",
    "num",
    "sector_2",
    "sector_3",
    "sector_4",
    "Revenue/employee",
    "Fixed_assets/employee",
    "EBITDA/cash_flow",
    YEAR_COL,
    LABEL_YEAR_COL,
    # firm-level snapshot from data extraction time: look-ahead leakage
    "operational_status",
]


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in COLUMNS_TO_DROP]


def prepare_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> SplitData:
    X = df[feature_columns(df)]
    y = df[TARGET_COL].to_numpy(dtype=int)

    X_train = X.iloc[train_idx].astype(float)
    medians = X_train.median().fillna(0.0)
    scaler = StandardScaler().fit(X_train.fillna(medians))

    def transform(idx: np.ndarray) -> pd.DataFrame:
        part = X.iloc[idx].astype(float).fillna(medians)
        return pd.DataFrame(scaler.transform(part), columns=X.columns)

    return SplitData(
        X_train=transform(train_idx),
        y_train=y[train_idx],
        X_val=transform(val_idx),
        y_val=y[val_idx],
        X_test=transform(test_idx),
        y_test=y[test_idx],
    )
