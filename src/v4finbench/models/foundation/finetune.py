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
    "none": "sample",
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
            n_finetune_ctx_plus_query_samples=config["batch_rows"],
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
            max_data_size=config["batch_rows"],
            n_estimators_finetune=1,
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
    start = time.perf_counter()
    X_train, y_train = apply_sampling(
        np.asarray(data.X_train, dtype=np.float32),
        data.y_train,
        SamplingConfig(
            strategy=sampling,
            minority_to_majority_ratio=config["ratio"],
            random_state=random_state,
            cache_dir=config.get("sampling_cache_dir"),
            max_rows=config.get("max_rows"),
        ),
    )
    sampling_seconds = time.perf_counter() - start
    X_val = np.asarray(data.X_val, dtype=np.float32)

    best: dict[str, Any] = {"score": -np.inf}
    trials: list[dict[str, Any]] = []
    for learning_rate in config["learning_rates"]:
        run_dir = output_dir / f"lr{learning_rate:g}"
        model = make_finetuner(estimator, learning_rate, config, random_state)
        start = time.perf_counter()
        model.fit(X_train, y_train, X_val=X_val, y_val=data.y_val, output_dir=run_dir)
        train_seconds = time.perf_counter() - start
        model = InContextModel(model)
        start = time.perf_counter()
        val_score = model.predict_proba(X_val)[:, 1]
        val_inference_seconds = time.perf_counter() - start
        score = average_precision_score(data.y_val, val_score)
        trials.append(
            {
                "learning_rate": learning_rate,
                "validation_pr_auc": score,
                "best_epoch": best_checkpoint(run_dir)[0],
                "train_seconds": round(train_seconds, 2),
                "val_inference_seconds": round(val_inference_seconds, 2),
            }
        )
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
                "trial": trials[-1],
            }
        else:
            shutil.rmtree(run_dir, ignore_errors=True)

    threshold = find_best_f1_threshold(data.y_val, best["val_score"])
    validation_f1 = binary_classification_metrics(
        data.y_val, best["val_score"], threshold
    )["f1"]
    start = time.perf_counter()
    test_score = best["model"].predict_proba(data.X_test)[:, 1]
    test_inference_seconds = time.perf_counter() - start

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
            "sampling_seconds": round(sampling_seconds, 2),
            "train_seconds": best["trial"]["train_seconds"],
            "val_inference_seconds": best["trial"]["val_inference_seconds"],
            "test_inference_seconds": round(test_inference_seconds, 2),
            "test_rows_per_second": round(len(test_score) / test_inference_seconds),
            "tuning_train_seconds": round(sum(t["train_seconds"] for t in trials), 2),
            "tuning_val_inference_seconds": round(
                sum(t["val_inference_seconds"] for t in trials), 2
            ),
            "trials": trials,
        },
        test_idx=split.test_idx,
        test_labels=data.y_test,
        test_scores=test_score,
        unseen_firm=unseen,
    )
