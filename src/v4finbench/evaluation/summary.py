import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from v4finbench.data.schema import COUNTRY_COL, FIRM_COL
from v4finbench.evaluation.metrics import FPR_BUDGETS

METRICS = [
    "roc_auc",
    *[f"recall_at_fpr_{budget:g}" for budget in FPR_BUDGETS],
    "pr_auc",
    "roc_auc_pooled",
    "precision",
    "recall",
    "f1",
]


@dataclass(frozen=True)
class _Ranked:
    y: np.ndarray
    y_pred: np.ndarray
    firm: np.ndarray
    tie_ends: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "_Ranked":
        order = np.argsort(-frame["score"].to_numpy(), kind="mergesort")
        score = frame["score"].to_numpy()[order]
        return cls(
            y=frame["y_true"].to_numpy(dtype=float)[order],
            y_pred=frame["y_pred"].to_numpy(dtype=bool)[order],
            firm=frame["_firm"].to_numpy()[order],
            tie_ends=np.r_[np.flatnonzero(np.diff(score)), len(score) - 1],
        )

    def curve(self, firm_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weight = firm_weight[self.firm]
        tp = np.cumsum(weight * self.y)[self.tie_ends]
        fp = np.cumsum(weight * (1 - self.y))[self.tie_ends]
        return tp, fp

    def counts(self, firm_weight: np.ndarray) -> np.ndarray:
        weight = firm_weight[self.firm]
        positive = self.y == 1
        return np.array(
            [
                np.sum(weight[self.y_pred & positive]),
                np.sum(weight[self.y_pred & ~positive]),
                np.sum(weight[~self.y_pred & positive]),
            ]
        )


@dataclass(frozen=True)
class _ModelScores:
    tasks: list[_Ranked]
    strata: list[_Ranked]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "_ModelScores":
        return cls(
            tasks=[_Ranked.from_frame(g) for _, g in frame.groupby("test_year")],
            strata=[
                _Ranked.from_frame(g)
                for _, g in frame.groupby(["test_year", COUNTRY_COL])
            ],
        )

    def metrics(self, firm_weight: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            strata = [s.curve(firm_weight) for s in self.strata]
            tasks = [t.curve(firm_weight) for t in self.tasks]
            pairs = np.array([tp[-1] * fp[-1] for tp, fp in strata])
            positives = np.array([tp[-1] if fp[-1] else 0.0 for tp, fp in strata])
            roc_auc = _weighted(strata, pairs, _auc)
            recalls = [
                _weighted(strata, positives, lambda tp, fp, b=b: _recall(tp, fp, b))
                for b in FPR_BUDGETS
            ]
            pr_auc = np.nanmean([_ap(tp, fp) for tp, fp in tasks])
            roc_auc_pooled = np.nanmean([_auc(tp, fp) for tp, fp in tasks])
            tp, fp, fn = np.sum([t.counts(firm_weight) for t in self.tasks], axis=0)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn)
            f1 = 2 * tp / (2 * tp + fp + fn)
        return np.array(
            [roc_auc, *recalls, pr_auc, roc_auc_pooled, precision, recall, f1]
        )


def _weighted(curves: list, weights: np.ndarray, metric) -> float:
    used = weights > 0
    values = [metric(tp, fp) for (tp, fp), u in zip(curves, used, strict=True) if u]
    return np.dot(values, weights[used]) / weights[used].sum()


def _rates(tp: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.r_[0.0, tp / tp[-1]], np.r_[0.0, fp / fp[-1]]


def _auc(tp: np.ndarray, fp: np.ndarray) -> float:
    tpr, fpr = _rates(tp, fp)
    return np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)


def _recall(tp: np.ndarray, fp: np.ndarray, budget: float) -> float:
    tpr, fpr = _rates(tp, fp)
    return tpr[fpr <= budget].max()


def _ap(tp: np.ndarray, fp: np.ndarray) -> float:
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    return np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * precision)


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
            model: _ModelScores.from_frame(model_scores)
            for model, model_scores in horizon_scores.groupby("model")
        }
        point = {model: m.metrics(np.ones(len(firms))) for model, m in models.items()}
        boot = {model: np.empty((n_boot, len(METRICS))) for model in models}
        for b in range(n_boot):
            firm_weight = np.bincount(
                rng.integers(0, len(firms), len(firms)), minlength=len(firms)
            ).astype(float)
            for model, m in models.items():
                boot[model][b] = m.metrics(firm_weight)

        for model, m in models.items():
            low, high = np.nanquantile(boot[model], quantiles, axis=0)
            row = {"model": model, "horizon": horizon, "n_tasks": len(m.tasks)}
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
