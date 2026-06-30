import json

import pandas as pd
import pytest

from v4finbench.evaluation.results import (
    aggregate_best_epochs,
    aggregate_best_epochs_by_horizon,
    collect_best_epoch_by_horizon_files,
    collect_best_epoch_files,
)


def test_collect_and_aggregate_best_epoch_files(tmp_path) -> None:
    for fold, f1 in [(0, 0.2), (1, 0.4)]:
        path = tmp_path / "h0" / f"fold_{fold}" / "best_epoch.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "model": "tabpfn_finetuned",
                    "horizon": 0,
                    "fold": fold,
                    "epoch": fold,
                    "validation_f1": f1,
                    "f1": f1,
                    "roc_auc": 0.8 + f1,
                }
            ),
            encoding="utf-8",
        )

    collected = collect_best_epoch_files(tmp_path)
    _, summary = aggregate_best_epochs(
        tmp_path,
        tmp_path / "best_epochs.csv",
        tmp_path / "summary.csv",
    )

    assert len(collected) == 2
    assert summary.iloc[0]["f1_mean"] == pytest.approx(0.3)
    assert (tmp_path / "best_epochs.csv").exists()
    assert (tmp_path / "summary.csv").exists()


def test_collect_and_aggregate_best_epoch_by_horizon_files(tmp_path) -> None:
    for fold, h0_f1, h1_f1 in [(0, 0.2, 0.4), (1, 0.6, 0.8)]:
        path = tmp_path / "h_all" / f"fold_{fold}" / "best_epoch_by_horizon.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "model": "tabpfn_finetuned",
                    "horizon": 0,
                    "fold": fold,
                    "epoch": 1,
                    "validation_f1": h0_f1,
                    "f1": h0_f1,
                    "roc_auc": 0.8,
                },
                {
                    "model": "tabpfn_finetuned",
                    "horizon": 1,
                    "fold": fold,
                    "epoch": 1,
                    "validation_f1": h1_f1,
                    "f1": h1_f1,
                    "roc_auc": 0.9,
                },
            ]
        ).to_csv(path, index=False)

    collected = collect_best_epoch_by_horizon_files(tmp_path)
    _, summary, means = aggregate_best_epochs_by_horizon(
        tmp_path,
        tmp_path / "best_epochs_by_horizon.csv",
        tmp_path / "summary_by_horizon.csv",
        tmp_path / "mean_metrics_by_horizon.csv",
    )

    assert len(collected) == 4
    h0 = means[means["horizon"] == 0].iloc[0]
    h1 = summary[summary["horizon"] == 1].iloc[0]
    assert h0["f1"] == pytest.approx(0.4)
    assert h0["validation_f1"] == pytest.approx(0.4)
    assert h0["n_folds"] == 2
    assert h1["f1_mean"] == pytest.approx(0.6)
    assert (tmp_path / "best_epochs_by_horizon.csv").exists()
    assert (tmp_path / "summary_by_horizon.csv").exists()
    assert (tmp_path / "mean_metrics_by_horizon.csv").exists()
