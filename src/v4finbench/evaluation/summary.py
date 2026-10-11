import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from v4finbench.data.schema import FIRM_COL
from v4finbench.evaluation.metrics import FPR_BUDGETS

RANKING_METRICS = [
    "roc_auc",
    "pr_auc",
    *[f"recall_at_fpr_{budget:g}" for budget in FPR_BUDGETS],
]
COUNT_METRICS = ["precision", "recall", "f1"]
METRICS = RANKING_METRICS + COUNT_METRICS


@dataclass(frozen=True)
class _RankedYear:
    y: np.ndarray
    y_pred: np.ndarray
    firm: np.ndarray
    tie_ends: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, firm: np.ndarray) -> "_RankedYear":
        order = np.argsort(-frame["score"].to_numpy(), kind="mergesort")
        score = frame["score"].to_numpy()[order]
        return cls(
            y=frame["y_true"].to_numpy(dtype=float)[order],
            y_pred=frame["y_pred"].to_numpy(dtype=bool)[order],
            firm=firm[order],
            tie_ends=np.r_[np.flatnonzero(np.diff(score)), len(score) - 1],
        )

    def ranking_metrics(self, weight: np.ndarray) -> np.ndarray:
        tp = np.cumsum(weight * self.y)[self.tie_ends]
        fp = np.cumsum(weight * (1 - self.y))[self.tie_ends]
        tpr = np.r_[0.0, tp / tp[-1]]
        fpr = np.r_[0.0, fp / fp[-1]]
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
        return np.array(
            [
                np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2),
                np.sum(np.diff(tpr) * precision),
                *[tpr[fpr <= budget].max() for budget in FPR_BUDGETS],
            ]
        )

    def counts(self, weight: np.ndarray) -> np.ndarray:
        positive = self.y == 1
        return np.array(
            [
                np.sum(weight[self.y_pred & positive]),
                np.sum(weight[self.y_pred & ~positive]),
                np.sum(weight[~self.y_pred & positive]),
            ]
        )


def _model_metrics(years: list[_RankedYear], firm_weight: np.ndarray) -> np.ndarray:
    weighted = [(year, firm_weight[year.firm]) for year in years]
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ranking = np.nanmean([year.ranking_metrics(w) for year, w in weighted], axis=0)
        tp, fp, fn = np.sum([year.counts(w) for year, w in weighted], axis=0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return np.r_[ranking, precision, recall, f1]


def evaluate_scores(
    scores: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    quantiles = [alpha / 2, 1 - alpha / 2]
    summary_rows, paired_rows = [], []

    for horizon, horizon_scores in scores.groupby("horizon"):
        firm, firms = pd.factorize(horizon_scores[FIRM_COL])
        horizon_scores = horizon_scores.assign(_firm=firm)
        models = {
            model: [
                _RankedYear.from_frame(year, year["_firm"].to_numpy())
                for _, year in model_scores.groupby("test_year")
            ]
            for model, model_scores in horizon_scores.groupby("model")
        }
        point = {
            model: _model_metrics(years, np.ones(len(firms)))
            for model, years in models.items()
        }
        boot = {model: np.empty((n_boot, len(METRICS))) for model in models}
        for b in range(n_boot):
            firm_weight = np.bincount(
                rng.integers(0, len(firms), len(firms)), minlength=len(firms)
            ).astype(float)
            for model, years in models.items():
                boot[model][b] = _model_metrics(years, firm_weight)

        for model, years in models.items():
            low, high = np.nanquantile(boot[model], quantiles, axis=0)
            row = {"model": model, "horizon": horizon, "n_test_years": len(years)}
            for i, metric in enumerate(METRICS):
                row |= {
                    metric: point[model][i],
                    f"{metric}_low": low[i],
                    f"{metric}_high": high[i],
                }
            summary_rows.append(row)

        for a, b in combinations(sorted(models), 2):
            diff = boot[a] - boot[b]
            low, high = np.nanquantile(diff, quantiles, axis=0)
            for i, metric in enumerate(METRICS):
                paired_rows.append(
                    {
                        "horizon": horizon,
                        "model_a": a,
                        "model_b": b,
                        "metric": metric,
                        "difference": point[a][i] - point[b][i],
                        "low": low[i],
                        "high": high[i],
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)
