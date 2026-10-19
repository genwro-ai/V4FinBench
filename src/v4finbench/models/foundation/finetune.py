import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from tabicl import FinetunedTabICLClassifier
from tabpfn.finetuning import FinetunedTabPFNClassifier

from v4finbench.data.preprocessing import prepare_split
from v4finbench.data.splits import TemporalSplit
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.protocol import evaluate_test_scores, unseen_firm_mask
from v4finbench.evaluation.results import RunResult
from v4finbench.evaluation.thresholds import find_best_f1_threshold
from v4finbench.models.foundation.in_context import InContextModel
from v4finbench.models.foundation.sampling import (
    SamplingConfig,
    apply_sampling,
    torch_device,
)

SAMPLING_NAMES = {
    "none": "full",
    "undersample": "undersample",
    "prototype_undersampling": "proto",
}


def finetuned_model_name(estimator: str, sampling: str) -> str:
    return f"{estimator}_finetuned_{SAMPLING_NAMES[sampling]}"


def make_finetuner(
    estimator: str, learning_rate: float, config: dict[str, Any], random_state: int
):
    device = str(torch_device(config.get("device", "auto")))
    if estimator == "tabpfn":
        return FinetunedTabPFNClassifier(
            device=device,
            epochs=config["epochs"],
            learning_rate=learning_rate,
            early_stopping=True,
            early_stopping_patience=config["patience"],
            eval_metric=config["eval_metric"],
            save_checkpoint_interval=None,
            random_state=random_state,
        )
    if estimator == "tabicl":
        return FinetunedTabICLClassifier(
            device=device,
            epochs=config["epochs"],
            learning_rate=learning_rate,
            early_stopping=True,
            patience=config["patience"],
            eval_metric=config["eval_metric"],
            save_interval=0,
            random_state=random_state,
        )
    raise ValueError(f"Unknown estimator: {estimator}")


def best_checkpoint(run_dir: Path) -> tuple[int, str]:
    checkpoints = [*run_dir.glob("checkpoint_*_best.pth"), *run_dir.glob("best.ckpt")]
    if not checkpoints:
        return 0, "pretrained"
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    return int(checkpoint["epoch"]), str(checkpoints[0])


def run_finetune(
    df: pd.DataFrame,
    split: TemporalSplit,
    estimator: str,
    sampling: str,
    config: dict[str, Any],
    output_dir: str | Path,
    random_state: int = 42,
) -> RunResult:
    output_dir = Path(output_dir)
    data = prepare_split(df, split.train_idx, split.val_idx, split.test_idx)
    X_train, y_train = apply_sampling(
        np.asarray(data.X_train, dtype=np.float32),
        data.y_train,
        SamplingConfig(
            strategy=sampling,
            minority_to_majority_ratio=config["ratio"],
            random_state=random_state,
            cache_dir=config.get("sampling_cache_dir"),
        ),
    )
    X_val = np.asarray(data.X_val, dtype=np.float32)

    best: dict[str, Any] = {"score": -np.inf}
    fit_start = time.perf_counter()
    for learning_rate in config["learning_rates"]:
        run_dir = output_dir / f"lr{learning_rate:g}"
        model = make_finetuner(estimator, learning_rate, config, random_state)
        model.fit(X_train, y_train, X_val=X_val, y_val=data.y_val, output_dir=run_dir)
        model = InContextModel(model)
        val_score = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(data.y_val, val_score)
        if score > best["score"]:
            if "run_dir" in best:
                shutil.rmtree(best["run_dir"], ignore_errors=True)
            best = {
                "score": score,
                "model": model,
                "learning_rate": learning_rate,
                "run_dir": run_dir,
                "val_score": val_score,
                "checkpoint": best_checkpoint(run_dir),
            }
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
    fit_seconds = time.perf_counter() - fit_start

    threshold = find_best_f1_threshold(data.y_val, best["val_score"])
    validation_f1 = binary_classification_metrics(
        data.y_val, best["val_score"], threshold
    )["f1"]
    predict_start = time.perf_counter()
    test_score = best["model"].predict_proba(data.X_test)[:, 1]
    predict_seconds = time.perf_counter() - predict_start

    unseen = unseen_firm_mask(df, split.train_idx, split.test_idx)
    return RunResult(
        model=finetuned_model_name(estimator, sampling),
        horizon=split.horizon,
        test_year=split.test_year,
        validation_f1=validation_f1,
        metrics=evaluate_test_scores(data.y_test, test_score, threshold, unseen),
        details={
            "learning_rate": best["learning_rate"],
            "best_epoch": best["checkpoint"][0],
            "validation_pr_auc": best["score"],
            "weights": best["checkpoint"][1],
            "n_train_rows": int(len(y_train)),
            "fit_seconds": round(fit_seconds, 2),
            "predict_seconds": round(predict_seconds, 4),
        },
        test_idx=split.test_idx,
        test_labels=data.y_test,
        test_scores=test_score,
        unseen_firm=unseen,
    )
