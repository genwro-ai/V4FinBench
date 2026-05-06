import json

import pytest

from v4finbench.evaluation.results import (
    aggregate_best_epochs,
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
