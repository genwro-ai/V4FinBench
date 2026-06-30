import json
from pathlib import Path

import pandas as pd

DEFAULT_GROUP_COLUMNS = ["model", "horizon"]
DEFAULT_METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]
DEFAULT_MEAN_METRIC_COLUMNS = ["validation_f1", *DEFAULT_METRIC_COLUMNS]


def aggregate_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str] | None = None,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    group_columns = group_columns or DEFAULT_GROUP_COLUMNS
    metric_columns = metric_columns or [
        col for col in DEFAULT_METRIC_COLUMNS if col in metrics.columns
    ]
    if not metric_columns:
        raise ValueError("No metric columns found to aggregate.")

    grouped = metrics.groupby(group_columns, dropna=False)
    summary = grouped[metric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    if "fold" in metrics.columns:
        summary["n_folds"] = grouped["fold"].nunique()
    else:
        summary["n_folds"] = grouped.size()
    return summary.reset_index()


def aggregate_metrics_file(
    input_path: str | Path,
    output_path: str | Path,
    group_columns: list[str] | None = None,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    metrics = pd.read_csv(input_path)
    summary = aggregate_metrics(metrics, group_columns, metric_columns)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return summary


def mean_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str] | None = None,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    group_columns = group_columns or DEFAULT_GROUP_COLUMNS
    metric_columns = metric_columns or [
        col for col in DEFAULT_MEAN_METRIC_COLUMNS if col in metrics.columns
    ]
    if not metric_columns:
        raise ValueError("No metric columns found to aggregate.")

    grouped = metrics.groupby(group_columns, dropna=False)
    summary = grouped[metric_columns].mean()
    if "fold" in metrics.columns:
        summary.insert(0, "n_folds", grouped["fold"].nunique())
    else:
        summary.insert(0, "n_folds", grouped.size())
    return summary.reset_index()


def collect_best_epoch_files(root: str | Path) -> pd.DataFrame:
    root_path = Path(root)
    rows = []
    for path in sorted(root_path.glob("h*/fold_*/best_epoch.json")):
        with path.open("r", encoding="utf-8") as file:
            row = json.load(file)
        row["source_file"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No best_epoch.json files found under {root_path}")
    return pd.DataFrame(rows)


def collect_best_epoch_by_horizon_files(root: str | Path) -> pd.DataFrame:
    root_path = Path(root)
    frames = []
    for path in sorted(root_path.glob("h*/fold_*/best_epoch_by_horizon.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No best_epoch_by_horizon.csv files found under {root_path}"
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_best_epochs(
    root: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_epochs = collect_best_epoch_files(root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_epochs.to_csv(output, index=False)

    summary = aggregate_metrics(best_epochs)
    if summary_path is not None:
        summary_output = Path(summary_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_output, index=False)
    return best_epochs, summary


def aggregate_best_epochs_by_horizon(
    root: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    means_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_epochs = collect_best_epoch_by_horizon_files(root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_epochs.to_csv(output, index=False)

    summary = aggregate_metrics(best_epochs)
    if summary_path is not None:
        summary_output = Path(summary_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_output, index=False)

    means = mean_metrics(best_epochs)
    if means_path is not None:
        means_output = Path(means_path)
        means_output.parent.mkdir(parents=True, exist_ok=True)
        means.to_csv(means_output, index=False)
    return best_epochs, summary, means
