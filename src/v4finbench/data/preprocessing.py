from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMNS_TO_DROP = [
    "company",
    "industry",
    "link",
    "num",
    "emis_id",
    "sector_2",
    "sector_3",
    "sector_4",
    "Revenue/employee",
    "Fixed_assets/employee",
    "EBITDA/cash_flow",
]


@dataclass(frozen=True)
class PreparedFeatures:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "main_label",
    group_col: str = "company",
    columns_to_drop: list[str] | None = None,
) -> PreparedFeatures:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    drop_cols = [target_col, *(columns_to_drop or COLUMNS_TO_DROP)]
    available_drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=available_drop_cols)
    y = df[target_col].astype(int)
    groups = df[group_col] if group_col in df.columns else None
    return PreparedFeatures(X=X, y=y, groups=groups)


@dataclass
class TabularPreprocessor:
    numeric_cols: list[str] | None = None
    medians: pd.Series | None = None
    scaler: StandardScaler | None = None

    def fit(self, X_train: pd.DataFrame) -> TabularPreprocessor:
        numeric_cols = list(X_train.select_dtypes(include=["number", "bool"]).columns)
        numeric_train = X_train[numeric_cols].apply(pd.to_numeric, errors="coerce")
        medians = numeric_train.median()
        # Columns with no observed training values (e.g. year-over-year
        # growth features when the training window is the panel's first
        # year) have NaN medians. Impute them as constant zero so every
        # split keeps the full feature width: leaving them NaN crashes
        # NaN-intolerant models and makes TabPFN's internal pipeline drop
        # the columns at fit time, breaking the train/predict feature-count
        # contract. A no-op whenever every column has observed values.
        medians = medians.fillna(0.0)
        scaler = StandardScaler()
        scaler.fit(numeric_train.fillna(medians))

        self.numeric_cols = numeric_cols
        self.medians = medians
        self.scaler = scaler
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.numeric_cols is None or self.medians is None or self.scaler is None:
            raise RuntimeError("TabularPreprocessor must be fitted before transform.")

        transformed = X.copy()
        numeric = transformed[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(self.medians)
        transformed[self.numeric_cols] = self.scaler.transform(numeric)
        return transformed

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)


def preprocess_train_val_test(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TabularPreprocessor]:
    preprocessor = TabularPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_val_processed, X_test_processed, preprocessor
