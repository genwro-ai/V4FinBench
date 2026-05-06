import pandas as pd

from v4finbench.data.preprocessing import (
    COLUMNS_TO_DROP,
    TabularPreprocessor,
    preprocess_train_val_test,
    split_features_target,
)


def test_split_features_target_drops_non_feature_columns() -> None:
    df = pd.DataFrame(
        {
            "company": ["a", "b"],
            "industry": ["x", "y"],
            "feature": [1.0, 2.0],
            "main_label": [0, 1],
        }
    )

    prepared = split_features_target(df)

    assert prepared.X.columns.tolist() == ["feature"]
    assert prepared.y.tolist() == [0, 1]
    assert prepared.groups.tolist() == ["a", "b"]


def test_default_drop_columns_do_not_include_legacy_article_columns() -> None:
    assert "available_articles" not in COLUMNS_TO_DROP
    assert "downloaded_articles" not in COLUMNS_TO_DROP
    assert "text_analysis" not in COLUMNS_TO_DROP


def test_preprocessor_uses_train_medians_only() -> None:
    X_train = pd.DataFrame({"feature": [1.0, None, 3.0]})
    X_val = pd.DataFrame({"feature": [None]})

    preprocessor = TabularPreprocessor().fit(X_train)
    transformed = preprocessor.transform(X_val)

    assert preprocessor.medians["feature"] == 2.0
    assert transformed["feature"].iloc[0] == 0.0


def test_preprocess_train_val_test_preserves_non_numeric_columns() -> None:
    X_train = pd.DataFrame({"feature": [1.0, 3.0], "category": ["a", "b"]})
    X_val = pd.DataFrame({"feature": [2.0], "category": ["a"]})
    X_test = pd.DataFrame({"feature": [4.0], "category": ["b"]})

    _, X_val_processed, _, _ = preprocess_train_val_test(X_train, X_val, X_test)

    assert X_val_processed["category"].tolist() == ["a"]
