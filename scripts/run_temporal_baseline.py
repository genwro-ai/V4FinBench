import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

from v4finbench.data.preprocessing import (
    COLUMNS_TO_DROP,
    preprocess_train_val_test,
    split_features_target,
)
from v4finbench.data.schema import HORIZON_FILES
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.thresholds import find_best_f1_threshold
from v4finbench.models.baselines import (
    iter_param_candidates,
    make_baseline_model,
    predict_positive_score,
)


TEMPORAL_MODELS = ["logistic_regression", "xgboost", "catboost"]

DROP = [*COLUMNS_TO_DROP, "year"]

FPR_BUDGETS = (0.001, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/temporal_splits"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/generated/temporal_baselines"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/baselines/standard.yaml"))
    parser.add_argument("--models", default="all",
                        help=f"'all' (= {','.join(TEMPORAL_MODELS)}) or a "
                             "comma-separated subset.")
    parser.add_argument("--horizon", default="all", help="'all' or one of 0..5")
    parser.add_argument("--test-year", default="all",
                        help="'all' or a single test year, e.g. 2015")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Randomly sample at most this many grid candidates.")
    parser.add_argument("--target-col", default="main_label")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-scores", action="store_true",
                        help="Skip writing per-instance score files.")
    return parser.parse_args()


def load_idx(fold_dir: Path, origin: int, kind: str) -> np.ndarray:
    return np.loadtxt(fold_dir / f"split_{origin}_{kind}_idx.txt", dtype=int)


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float) -> float:
    if y_true.sum() == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    feasible = fpr <= max_fpr
    return float(tpr[feasible].max()) if feasible.any() else 0.0


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        grids = yaml.safe_load(file)["models"]

    models = TEMPORAL_MODELS if args.models == "all" else args.models.split(",")
    unknown = set(models) - set(grids)
    if unknown:
        raise ValueError(f"No hyperparameter grid for: {sorted(unknown)}")
    horizons = sorted(HORIZON_FILES) if args.horizon == "all" else [int(args.horizon)]

    args.out.mkdir(parents=True, exist_ok=True)
    scores_dir = args.out / "scores"
    if not args.no_scores:
        scores_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.out / "metrics.csv"

    rows = []
    for horizon in horizons:
        df = pd.read_parquet(args.data_dir / HORIZON_FILES[horizon])
        prepared = split_features_target(df, target_col=args.target_col,
                                         columns_to_drop=DROP)
        companies = df["company"].to_numpy()
        y_all = prepared.y.to_numpy()

        horizon_dir = args.splits_dir / f"h{horizon}"
        meta = json.loads((horizon_dir / "metadata.json").read_text())
        origins = sorted(
            int(p.stem.split("_")[1])
            for p in horizon_dir.glob("split_*_train_idx.txt")
        )

        for origin in origins:
            test_year = int(meta["splits"][str(origin)]["origin"])
            if args.test_year != "all" and test_year != int(args.test_year):
                continue

            tr = load_idx(horizon_dir, origin, "train")
            va = load_idx(horizon_dir, origin, "val")
            te = load_idx(horizon_dir, origin, "test")

            X_tr, X_va, X_te, _ = preprocess_train_val_test(
                prepared.X.iloc[tr], prepared.X.iloc[va], prepared.X.iloc[te]
            )
            y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]

            train_companies = set(companies[tr])
            unseen = np.array([c not in train_companies for c in companies[te]])

            for model_name in models:
                best_model, best_params = None, None
                best_threshold, best_val_f1 = 0.5, -np.inf
                candidates = iter_param_candidates(
                    grids[model_name], args.max_candidates, args.random_state
                )
                n_candidates = 0
                grid_start = time.perf_counter()
                for params in candidates:
                    n_candidates += 1
                    model = make_baseline_model(model_name, params,
                                                args.random_state)
                    model.fit(X_tr, y_tr)
                    val_score = predict_positive_score(model, X_va)
                    threshold = find_best_f1_threshold(y_va, val_score)
                    val_f1 = binary_classification_metrics(
                        y_va, val_score, threshold)["f1"]
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_model, best_params = model, params
                        best_threshold = threshold
                grid_seconds = time.perf_counter() - grid_start

                if best_model is None:
                    raise RuntimeError(f"No candidate succeeded: {model_name}")

                predict_start = time.perf_counter()
                p_te = predict_positive_score(best_model, X_te)
                predict_seconds = time.perf_counter() - predict_start
                m = binary_classification_metrics(y_te, p_te, best_threshold)
                m_unseen = binary_classification_metrics(
                    y_te[unseen], p_te[unseen], best_threshold
                ) if unseen.any() and y_te[unseen].sum() > 0 else {}

                row = {
                    "model": model_name, "horizon": horizon,
                    "origin": origin, "test_year": test_year,
                    "train_pos": int(y_tr.sum()), "test_pos": int(y_te.sum()),
                    "test_n": int(len(te)),
                    "validation_f1": float(best_val_f1),
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["average_precision"],
                    "f1": m["f1"], "precision": m["precision"],
                    "recall": m["recall"], "threshold": m["threshold"],
                    "unseen_frac": float(unseen.mean()),
                    "unseen_roc_auc": m_unseen.get("roc_auc", float("nan")),
                    "unseen_pr_auc": m_unseen.get("average_precision",
                                                  float("nan")),
                    "n_candidates": n_candidates,
                    "grid_seconds": round(grid_seconds, 2),
                    "predict_seconds": round(predict_seconds, 4),
                    "predict_rows_per_s": round(len(te) / predict_seconds)
                    if predict_seconds > 0 else float("nan"),
                }
                for budget in FPR_BUDGETS:
                    row[f"recall_at_fpr_{budget:g}"] = recall_at_fpr(
                        y_te, p_te, budget)
                rows.append(row)
                pd.DataFrame([row]).to_csv(
                    metrics_csv, mode="a", header=not metrics_csv.exists(),
                    index=False,
                )
                print(f"{model_name:>20s} h={horizon} Y={test_year}: "
                      f"ROC-AUC={m['roc_auc']:.3f} "
                      f"PR-AUC={m['average_precision']:.3f} F1={m['f1']:.3f} "
                      f"R@FPR1%={row['recall_at_fpr_0.01']:.3f}", flush=True)

                if not args.no_scores:
                    pd.DataFrame({
                        "row_idx": te,
                        "y_true": y_te.astype(int),
                        "score": p_te,
                        "unseen_company": unseen,
                    }).to_parquet(
                        scores_dir / f"{model_name}_h{horizon}_Y{test_year}.parquet",
                        index=False,
                    )

    # Re-read the append-only CSV and deduplicate on the cell key so that
    # re-running any subset of cells replaces earlier rows instead of
    # accumulating duplicates. metrics.json is rebuilt from the deduplicated
    # full history, not just this invocation.
    all_rows = pd.read_csv(metrics_csv)
    all_rows = all_rows.drop_duplicates(
        subset=["model", "horizon", "test_year"], keep="last"
    )
    all_rows.to_csv(metrics_csv, index=False)
    (args.out / "metrics.json").write_text(
        all_rows.to_json(orient="records", indent=2))
    results = all_rows

    if {"model", "horizon"} <= set(results.columns) and len(results):
        full_grid = results[results.test_year <= 2015]
        if len(full_grid):
            print("\n=== Primary (full-grid, Y<=2015) ROC-AUC mean±sd ===")
            summary = (full_grid.groupby(["model", "horizon"])["roc_auc"]
                       .agg(["mean", "std"]).round(3))
            print(summary.to_string())
    print(f"\nWrote {metrics_csv}")


if __name__ == "__main__":
    main()
