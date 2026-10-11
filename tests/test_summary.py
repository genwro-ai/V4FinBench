import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from v4finbench.evaluation.metrics import recall_at_fpr
from v4finbench.evaluation.summary import evaluate_scores


def _scores(model: str, noise: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for test_year in (2013, 2014):
        y = (rng.random(400) < 0.1).astype(int)
        score = np.round(y + rng.normal(scale=noise, size=400), 1)
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "horizon": 1,
                    "test_year": test_year,
                    "emis_id": rng.integers(0, 150, 400),
                    "y_true": y,
                    "score": score,
                    "y_pred": score >= 0.5,
                }
            )
        )
    return pd.concat(frames)


def test_point_estimates_match_sklearn() -> None:
    scores = _scores("m", noise=0.8)
    summary, _ = evaluate_scores(scores, n_boot=5)
    years = [year for _, year in scores.groupby("test_year")]

    row = summary.iloc[0]
    assert row["roc_auc"] == pytest.approx(
        np.mean([roc_auc_score(y.y_true, y.score) for y in years])
    )
    assert row["pr_auc"] == pytest.approx(
        np.mean([average_precision_score(y.y_true, y.score) for y in years])
    )
    assert row["recall_at_fpr_0.01"] == pytest.approx(
        np.mean([recall_at_fpr(y.y_true, y.score, 0.01) for y in years])
    )
    tp = (scores.y_pred & (scores.y_true == 1)).sum()
    assert row["f1"] == pytest.approx(
        2 * tp / (scores.y_pred.sum() + scores.y_true.sum())
    )


def test_bootstrap_intervals_and_paired_differences() -> None:
    scores = pd.concat([_scores("good", noise=0.3), _scores("bad", noise=1.5)])
    summary, paired = evaluate_scores(scores, n_boot=200)

    for _, row in summary.iterrows():
        assert row["roc_auc_low"] <= row["roc_auc"] <= row["roc_auc_high"]
    auc = paired[paired.metric == "roc_auc"].iloc[0]
    assert (auc.model_a, auc.model_b) == ("bad", "good")
    assert auc.high < 0
