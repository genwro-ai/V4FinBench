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
        country = rng.integers(0, 2, 600)
        y = (rng.random(600) < 0.05 + 0.1 * country).astype(int)
        score = np.round(y + country + rng.normal(scale=noise, size=600), 1)
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "horizon": 1,
                    "test_year": test_year,
                    "emis_id": rng.integers(0, 200, 600),
                    "country": country,
                    "y_true": y,
                    "score": score,
                    "y_pred": score >= 1.5,
                }
            )
        )
    return pd.concat(frames)


def _stratified_auc(scores: pd.DataFrame) -> float:
    concordant = pairs = 0.0
    for _, s in scores.groupby(["test_year", "country"]):
        pos, neg = s.score[s.y_true == 1].to_numpy(), s.score[s.y_true == 0].to_numpy()
        diff = pos[:, None] - neg[None, :]
        concordant += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        pairs += diff.size
    return concordant / pairs


def test_point_estimates() -> None:
    scores = _scores("m", noise=0.8)
    row = evaluate_scores(scores, n_boot=5)[0].iloc[0]
    years = [year for _, year in scores.groupby("test_year")]

    assert row["roc_auc"] == pytest.approx(_stratified_auc(scores))
    assert row["roc_auc_pooled"] == pytest.approx(
        np.mean([roc_auc_score(y.y_true, y.score) for y in years])
    )
    assert row["pr_auc"] == pytest.approx(
        np.mean([average_precision_score(y.y_true, y.score) for y in years])
    )
    strata = [s for _, s in scores.groupby(["test_year", "country"])]
    assert row["recall_at_fpr_0.01"] == pytest.approx(
        sum(recall_at_fpr(s.y_true, s.score, 0.01) * s.y_true.sum() for s in strata)
        / scores.y_true.sum()
    )
    tp = (scores.y_pred & (scores.y_true == 1)).sum()
    assert row["f1"] == pytest.approx(
        2 * tp / (scores.y_pred.sum() + scores.y_true.sum())
    )


def test_stratified_auc_ignores_country_score_shift() -> None:
    row = evaluate_scores(_scores("m", noise=0.8), n_boot=5)[0].iloc[0]

    assert row["roc_auc"] < row["roc_auc_pooled"]


def test_bootstrap_intervals_and_paired_differences() -> None:
    scores = pd.concat([_scores("good", noise=0.3), _scores("bad", noise=1.5)])
    summary, paired = evaluate_scores(scores, n_boot=200)

    for _, row in summary.iterrows():
        assert row["roc_auc_low"] <= row["roc_auc"] <= row["roc_auc_high"]
    auc = paired[paired.metric == "roc_auc"].iloc[0]
    assert (auc.model_a, auc.model_b) == ("bad", "good")
    assert auc.high < 0
