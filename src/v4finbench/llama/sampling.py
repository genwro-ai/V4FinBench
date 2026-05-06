from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LlamaSplitConfig:
    train_size: int = 20_000
    test_size: int = 100_000
    min_positive_ratio_train: float = 0.10
    min_positive_ratio_test: float = 0.10
    seed: int = 42
    target_col: str = "main_label"
    deduplicate_cols: tuple[str, ...] = ("company", "year")


def prepare_source_dataframe(
    df: pd.DataFrame,
    *,
    target_col: str = "main_label",
    deduplicate_cols: tuple[str, ...] = ("company", "year"),
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    existing_dedup_cols = [col for col in deduplicate_cols if col in df.columns]
    if existing_dedup_cols:
        df = df.drop_duplicates(subset=existing_dedup_cols, keep="first")
    return df[df[target_col].isin([0, 1])].reset_index(drop=True)


def sample_balanced_split(
    df: pd.DataFrame,
    *,
    size: int,
    min_positive_ratio: float,
    target_col: str = "main_label",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample one split without replacement.

    The requested positive ratio is treated as a target, not a hard requirement.
    If a horizon has too few positives or negatives, the split uses all available
    rows from the limiting class and fills the rest from the other class.
    """
    if not 0 < min_positive_ratio < 1:
        raise ValueError("min_positive_ratio must be between 0 and 1.")
    if size <= 0:
        raise ValueError("size must be positive.")

    target_size = min(size, len(df))
    positives = df[df[target_col] == 1]
    negatives = df[df[target_col] == 0]
    desired_positive_count = int(round(target_size * min_positive_ratio))
    if len(positives) > 0:
        desired_positive_count = max(1, desired_positive_count)

    positive_count = min(desired_positive_count, len(positives))
    negative_count = min(target_size - positive_count, len(negatives))
    shortfall = target_size - positive_count - negative_count
    if shortfall > 0:
        extra_positive_count = min(shortfall, len(positives) - positive_count)
        positive_count += extra_positive_count

    if positive_count == 0 and negative_count == 0:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()

    pos_sample = positives.sample(n=positive_count, random_state=seed)
    neg_sample = negatives.sample(n=negative_count, random_state=seed + 1)
    split = (
        pd.concat([pos_sample, neg_sample])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    remaining = df.drop(index=pos_sample.index.union(neg_sample.index)).reset_index(
        drop=True
    )
    return split, remaining


def create_llama_train_test_splits(
    df: pd.DataFrame,
    config: LlamaSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = prepare_source_dataframe(
        df,
        target_col=config.target_col,
        deduplicate_cols=config.deduplicate_cols,
    )
    train_df, remaining = sample_balanced_split(
        source,
        size=config.train_size,
        min_positive_ratio=config.min_positive_ratio_train,
        target_col=config.target_col,
        seed=config.seed + 10,
    )
    test_df, _ = sample_balanced_split(
        remaining,
        size=config.test_size,
        min_positive_ratio=config.min_positive_ratio_test,
        target_col=config.target_col,
        seed=config.seed,
    )
    return train_df, test_df
