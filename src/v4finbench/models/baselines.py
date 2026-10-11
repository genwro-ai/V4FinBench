import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from v4finbench.data.preprocessing import SplitData, prepare_split
from v4finbench.data.splits import TemporalSplit
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.protocol import evaluate_test_scores, unseen_firm_mask
from v4finbench.evaluation.results import RunResult
from v4finbench.evaluation.thresholds import find_best_f1_threshold

MAX_TREES = 5000


def run_baseline(
    df: pd.DataFrame,
    split: TemporalSplit,
    model_name: str,
    config: dict[str, Any],
    random_state: int = 42,
    n_trials: int | None = None,
    output_dir: str | Path | None = None,
    progress: bool = False,
) -> RunResult:
    data = prepare_split(df, split.train_idx, split.val_idx, split.test_idx)
    model_config = config["models"][model_name]
    search = model_config.get("search", {})
    fixed = model_config.get("params", {})
    n_trials = (n_trials or config["n_trials"]) if search else 1

    best: dict[str, Any] = {"score": -np.inf}

    def objective(trial: optuna.Trial) -> float:
        params = {
            **fixed,
            **{name: suggest(trial, name, space) for name, space in search.items()},
        }
        model = fit_model(
            model_name, params, data, random_state, config["early_stopping_rounds"]
        )
        score = average_precision_score(data.y_val, positive_score(model, data.X_val))
        if score > best["score"]:
            best.update(score=score, model=model, params=params)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    fit_start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=progress)
    fit_seconds = time.perf_counter() - fit_start

    model = best["model"]
    val_score = positive_score(model, data.X_val)
    threshold = find_best_f1_threshold(data.y_val, val_score)
    validation_f1 = binary_classification_metrics(data.y_val, val_score, threshold)[
        "f1"
    ]

    predict_start = time.perf_counter()
    test_score = positive_score(model, data.X_test)
    predict_seconds = time.perf_counter() - predict_start

    unseen = unseen_firm_mask(df, split.train_idx, split.test_idx)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_dir / f"{model_name}.joblib")
        (output_dir / f"{model_name}.params.json").write_text(
            json.dumps({"params": best["params"], "threshold": threshold}, indent=2),
            encoding="utf-8",
        )

    return RunResult(
        model=model_name,
        horizon=split.horizon,
        test_year=split.test_year,
        validation_f1=validation_f1,
        metrics=evaluate_test_scores(data.y_test, test_score, threshold, unseen),
        details={
            "best_params": best["params"],
            "validation_pr_auc": best["score"],
            "n_trials": n_trials,
            "fit_seconds": round(fit_seconds, 2),
            "predict_seconds": round(predict_seconds, 4),
        },
        test_idx=split.test_idx,
        test_labels=data.y_test,
        test_scores=test_score,
        unseen_firm=unseen,
    )


def suggest(trial: optuna.Trial, name: str, space: dict[str, Any]) -> Any:
    if "choices" in space:
        index = trial.suggest_categorical(name, list(range(len(space["choices"]))))
        return space["choices"][index]
    low, high, log = space["low"], space["high"], space.get("log", False)
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(name, low, high, log=log)
    return trial.suggest_float(name, low, high, log=log)


def fit_model(
    model_name: str,
    params: dict[str, Any],
    data: SplitData,
    random_state: int,
    early_stopping_rounds: int,
):
    X, y, X_val, y_val = data.X_train, data.y_train, data.X_val, data.y_val
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, random_state=random_state, **params
        ).fit(X, y)
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=MAX_TREES,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=random_state,
            **params,
        )
        return model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping

        model = LGBMClassifier(
            n_estimators=MAX_TREES,
            metric="average_precision",
            subsample_freq=1,
            n_jobs=-1,
            verbosity=-1,
            random_state=random_state,
            **params,
        )
        return model.fit(
            X,
            y,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=MAX_TREES,
            eval_metric="PRAUC",
            od_type="Iter",
            od_wait=early_stopping_rounds,
            use_best_model=True,
            bootstrap_type="Bayesian",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_state,
            **params,
        )
        return model.fit(X, y, eval_set=(X_val, y_val))
    if model_name in ("realmlp", "tabm"):
        from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier

        estimator = (
            RealMLP_TD_Classifier if model_name == "realmlp" else TabM_D_Classifier
        )
        return estimator(random_state=random_state, **params).fit(X, y, X_val, y_val)
    raise ValueError(f"Unknown baseline model: {model_name}")


def positive_score(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
