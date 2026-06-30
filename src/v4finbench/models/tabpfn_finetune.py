import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.finetune_utils import clone_model_for_evaluation
from torchvision.ops import sigmoid_focal_loss
from tqdm.auto import tqdm

from v4finbench.data.preprocessing import (
    preprocess_train_val_test,
    split_features_target,
)
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.thresholds import find_best_f1_threshold
from v4finbench.models.tabpfn import (
    prototype_backend_from_mapping,
    select_context_samples,
    subsample_indices,
)
from v4finbench.sampling.strategies import SamplingConfig, apply_sampling


@dataclass(frozen=True)
class TabPFNFinetuneConfig:
    sampling_strategy: str = "prototype_undersampling"
    n_inference_context_samples: int = 10_000
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    minority_to_majority_ratio: float = 0.3
    random_state: int = 42
    device: str = "auto"
    prototype_backend: str = "cuml"
    model_path: str | None = None
    inference_precision: str = "auto"
    eval_batch_size: int | None = 8192
    epochs: int = 10
    learning_rate: float = 5e-6
    batch_size: int = 1024
    meta_batch_size: int = 1
    loss_function: str = "cross_entropy"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    save_checkpoints: bool = False
    save_best_checkpoint: bool = True


@dataclass(frozen=True)
class TabPFNFinetuneEpochResult:
    model: str
    horizon: int
    fold: int
    epoch: int
    sampling_strategy: str
    n_context_samples: int
    validation_f1: float
    metrics: dict[str, float]


ClassifierFactory = Callable[[TabPFNFinetuneConfig], Any]


def finetune_config_from_mapping(values: dict[str, Any]) -> TabPFNFinetuneConfig:
    finetuning = values.get("finetuning", {})
    return TabPFNFinetuneConfig(
        sampling_strategy=values.get("sampling_strategy", "prototype_undersampling"),
        n_inference_context_samples=int(
            values.get(
                "n_inference_context_samples",
                values.get("n_context_samples", 10_000),
            )
        ),
        max_train_samples=_optional_int(values.get("max_train_samples")),
        max_eval_samples=_optional_int(values.get("max_eval_samples")),
        minority_to_majority_ratio=float(values.get("minority_to_majority_ratio", 0.3)),
        random_state=int(values.get("random_seed", values.get("random_state", 42))),
        device=values.get("device", "auto"),
        prototype_backend=prototype_backend_from_mapping(values),
        model_path=values.get("model_path"),
        inference_precision=str(values.get("inference_precision", "auto")),
        eval_batch_size=_optional_int(values.get("eval_batch_size", 8192)),
        epochs=int(finetuning.get("epochs", values.get("epochs", 10))),
        learning_rate=float(
            finetuning.get("learning_rate", values.get("learning_rate", 5e-6))
        ),
        batch_size=int(finetuning.get("batch_size", values.get("batch_size", 1024))),
        meta_batch_size=int(
            finetuning.get("meta_batch_size", values.get("meta_batch_size", 1))
        ),
        loss_function=values.get("loss_function", "cross_entropy"),
        focal_alpha=float(values.get("focal_alpha", 0.25)),
        focal_gamma=float(values.get("focal_gamma", 2.0)),
        save_checkpoints=bool(values.get("save_checkpoints", False)),
        save_best_checkpoint=bool(values.get("save_best_checkpoint", True)),
    )


def finetune_evaluate_tabpfn(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon: int,
    fold: int,
    config: TabPFNFinetuneConfig | None = None,
    target_col: str = "main_label",
    output_dir: str | Path | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> list[TabPFNFinetuneEpochResult]:
    config = config or TabPFNFinetuneConfig()
    arrays = _prepare_arrays(df, train_idx, val_idx, test_idx, config, target_col)
    return _finetune_evaluate_prepared_arrays(
        arrays=arrays,
        horizon=horizon,
        fold=fold,
        config=config,
        output_dir=output_dir,
        classifier_factory=classifier_factory,
    )


def finetune_evaluate_tabpfn_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    fold: int,
    config: TabPFNFinetuneConfig | None = None,
    target_col: str = "main_label",
    output_dir: str | Path | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> list[TabPFNFinetuneEpochResult]:
    config = config or TabPFNFinetuneConfig()
    arrays = _prepare_arrays_from_split_frames(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        target_col=target_col,
    )
    return _finetune_evaluate_prepared_arrays(
        arrays=arrays,
        horizon=horizon,
        fold=fold,
        config=config,
        output_dir=output_dir,
        classifier_factory=classifier_factory,
    )


def finetune_evaluate_tabpfn_splits_by_horizon(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons: list[int],
    fold: int,
    config: TabPFNFinetuneConfig | None = None,
    target_col: str = "main_label",
    horizon_col: str = "prediction_horizon",
    joint_horizon: int = -1,
    output_dir: str | Path | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> tuple[list[TabPFNFinetuneEpochResult], list[TabPFNFinetuneEpochResult]]:
    config = config or TabPFNFinetuneConfig()
    arrays, train_horizons, val_horizons, test_horizons = (
        _prepare_arrays_and_horizon_labels_from_split_frames(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            horizons=horizons,
            config=config,
            target_col=target_col,
            horizon_col=horizon_col,
        )
    )
    return _finetune_evaluate_prepared_arrays_by_horizon(
        arrays=arrays,
        train_horizons=train_horizons,
        val_horizons=val_horizons,
        test_horizons=test_horizons,
        horizons=horizons,
        joint_horizon=joint_horizon,
        fold=fold,
        config=config,
        output_dir=output_dir,
        classifier_factory=classifier_factory,
    )


def _finetune_evaluate_prepared_arrays(
    arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    horizon: int,
    fold: int,
    config: TabPFNFinetuneConfig,
    output_dir: str | Path | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> list[TabPFNFinetuneEpochResult]:
    X_train, y_train, X_val, y_val, X_test, y_test = arrays

    classifier, classifier_config = (
        classifier_factory(config)
        if classifier_factory is not None
        else make_finetunable_tabpfn_classifier(config)
    )
    optimizer = _make_optimizer(classifier, config)
    dataloader = _make_finetuning_dataloader(classifier, X_train, y_train, config)

    results = []
    best_result = None
    for epoch in range(config.epochs + 1):
        if epoch > 0:
            _finetune_one_epoch(classifier, optimizer, dataloader, config, epoch)

        result = evaluate_finetuned_tabpfn(
            classifier=classifier,
            classifier_config=classifier_config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            horizon=horizon,
            fold=fold,
            epoch=epoch,
            config=config,
        )
        results.append(result)
        if output_dir is not None:
            append_finetune_metrics(output_dir, result)
            if config.save_best_checkpoint and _is_better_result(result, best_result):
                save_tabpfn_torch_state_file(output_dir, classifier, "best_epoch.pt")
                best_result = result
            if config.save_checkpoints:
                save_tabpfn_torch_state(output_dir, classifier, epoch)

    if output_dir is not None:
        write_best_epoch(output_dir, best_result or select_best_epoch(results))
    return results


def _finetune_evaluate_prepared_arrays_by_horizon(
    arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    train_horizons: np.ndarray,
    val_horizons: np.ndarray,
    test_horizons: np.ndarray,
    horizons: list[int],
    joint_horizon: int,
    fold: int,
    config: TabPFNFinetuneConfig,
    output_dir: str | Path | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> tuple[list[TabPFNFinetuneEpochResult], list[TabPFNFinetuneEpochResult]]:
    X_train, y_train, X_val, y_val, X_test, y_test = arrays

    classifier, classifier_config = (
        classifier_factory(config)
        if classifier_factory is not None
        else make_finetunable_tabpfn_classifier(config)
    )
    optimizer = _make_optimizer(classifier, config)
    dataloaders = _make_horizon_conditioned_finetuning_dataloaders(
        classifier,
        X_train,
        y_train,
        train_horizons,
        horizons,
        config,
    )

    joint_results = []
    horizon_results = []
    best_joint_result = None
    for epoch in range(config.epochs + 1):
        if epoch > 0:
            _finetune_one_horizon_conditioned_epoch(
                classifier,
                optimizer,
                dataloaders,
                config,
                epoch,
            )

        joint_result, epoch_horizon_results = evaluate_finetuned_tabpfn_by_horizon(
            classifier=classifier,
            classifier_config=classifier_config,
            X_train=X_train,
            y_train=y_train,
            train_horizons=train_horizons,
            X_val=X_val,
            y_val=y_val,
            val_horizons=val_horizons,
            X_test=X_test,
            y_test=y_test,
            test_horizons=test_horizons,
            horizons=horizons,
            joint_horizon=joint_horizon,
            fold=fold,
            epoch=epoch,
            config=config,
        )
        joint_results.append(joint_result)
        horizon_results.extend(epoch_horizon_results)
        if output_dir is not None:
            append_finetune_metrics(output_dir, joint_result)
            append_finetune_horizon_metrics(output_dir, epoch_horizon_results)
            if config.save_best_checkpoint:
                if _is_better_result(joint_result, best_joint_result):
                    save_tabpfn_torch_state_file(
                        output_dir,
                        classifier,
                        "best_epoch.pt",
                    )
                    best_joint_result = joint_result
            if config.save_checkpoints:
                save_tabpfn_torch_state(output_dir, classifier, epoch)

    if output_dir is not None:
        best = best_joint_result or select_best_epoch(joint_results)
        write_best_epoch(output_dir, best)
        write_best_epoch_by_horizon(
            output_dir,
            [result for result in horizon_results if result.epoch == best.epoch],
        )
    return joint_results, horizon_results


def evaluate_finetuned_tabpfn(
    classifier,
    classifier_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    horizon: int,
    fold: int,
    epoch: int,
    config: TabPFNFinetuneConfig,
) -> TabPFNFinetuneEpochResult:
    eval_classifier = clone_tabpfn_for_evaluation(classifier, classifier_config)
    X_context, y_context = select_context_samples(
        X_train,
        y_train,
        n_context_samples=config.n_inference_context_samples,
        random_state=config.random_state,
    )
    eval_classifier.fit(X_context, y_context)

    val_score = _predict_positive_proba(
        eval_classifier,
        X_val,
        config.eval_batch_size,
    )
    threshold = find_best_f1_threshold(y_val, val_score)
    val_metrics = binary_classification_metrics(y_val, val_score, threshold)

    test_score = _predict_positive_proba(
        eval_classifier,
        X_test,
        config.eval_batch_size,
    )
    metrics = binary_classification_metrics(y_test, test_score, threshold)
    return TabPFNFinetuneEpochResult(
        model="tabpfn_finetuned",
        horizon=horizon,
        fold=fold,
        epoch=epoch,
        sampling_strategy=config.sampling_strategy,
        n_context_samples=len(y_context),
        validation_f1=val_metrics["f1"],
        metrics=metrics,
    )


def evaluate_finetuned_tabpfn_by_horizon(
    classifier,
    classifier_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_horizons: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_horizons: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_horizons: np.ndarray,
    horizons: list[int],
    joint_horizon: int,
    fold: int,
    epoch: int,
    config: TabPFNFinetuneConfig,
) -> tuple[TabPFNFinetuneEpochResult, list[TabPFNFinetuneEpochResult]]:
    horizon_results = evaluate_finetuned_tabpfn_horizon_slices(
        classifier=classifier,
        classifier_config=classifier_config,
        horizons=horizons,
        fold=fold,
        epoch=epoch,
        config=config,
        X_train=X_train,
        y_train=y_train,
        train_horizons=train_horizons,
        X_val=X_val,
        y_val=y_val,
        val_horizons=val_horizons,
        X_test=X_test,
        y_test=y_test,
        test_horizons=test_horizons,
    )
    joint_result = macro_average_finetune_results(
        horizon_results,
        horizon=joint_horizon,
        fold=fold,
        epoch=epoch,
        sampling_strategy=config.sampling_strategy,
    )
    return joint_result, horizon_results


def evaluate_finetuned_tabpfn_horizon_slices(
    classifier,
    classifier_config: dict[str, Any],
    horizons: list[int],
    fold: int,
    epoch: int,
    config: TabPFNFinetuneConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_horizons: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_horizons: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_horizons: np.ndarray,
) -> list[TabPFNFinetuneEpochResult]:
    results = []
    for horizon in horizons:
        train_mask = train_horizons == horizon
        val_mask = val_horizons == horizon
        test_mask = test_horizons == horizon
        if not train_mask.any() or not val_mask.any() or not test_mask.any():
            raise ValueError(
                f"Cannot evaluate horizon {horizon}: "
                f"train rows={int(train_mask.sum())}, "
                f"validation rows={int(val_mask.sum())}, "
                f"test rows={int(test_mask.sum())}."
            )

        eval_classifier = clone_tabpfn_for_evaluation(classifier, classifier_config)
        X_context, y_context = select_context_samples(
            X_train[train_mask],
            y_train[train_mask],
            n_context_samples=config.n_inference_context_samples,
            random_state=config.random_state + horizon,
        )
        eval_classifier.fit(X_context, y_context)

        val_score = _predict_positive_proba(
            eval_classifier,
            X_val[val_mask],
            config.eval_batch_size,
        )
        test_score = _predict_positive_proba(
            eval_classifier,
            X_test[test_mask],
            config.eval_batch_size,
        )
        result = finetune_result_from_scores(
            horizon=horizon,
            fold=fold,
            epoch=epoch,
            sampling_strategy=config.sampling_strategy,
            n_context_samples=len(y_context),
            y_val=y_val[val_mask],
            val_score=val_score,
            y_test=y_test[test_mask],
            test_score=test_score,
        )
        result.metrics["validation_rows"] = int(val_mask.sum())
        result.metrics["test_rows"] = int(test_mask.sum())
        results.append(result)
    return results


def macro_average_finetune_results(
    results: list[TabPFNFinetuneEpochResult],
    horizon: int,
    fold: int,
    epoch: int,
    sampling_strategy: str,
) -> TabPFNFinetuneEpochResult:
    if not results:
        raise ValueError("Cannot macro-average an empty result list.")

    metric_keys = sorted({key for result in results for key in result.metrics})
    metrics = {}
    for key in metric_keys:
        values = [result.metrics[key] for result in results if key in result.metrics]
        if key in {"validation_rows", "test_rows"}:
            metrics[key] = int(np.sum(values))
        else:
            metrics[key] = float(np.nanmean(values))
    return TabPFNFinetuneEpochResult(
        model=results[0].model,
        horizon=horizon,
        fold=fold,
        epoch=epoch,
        sampling_strategy=sampling_strategy,
        n_context_samples=int(np.sum([result.n_context_samples for result in results])),
        validation_f1=float(np.mean([result.validation_f1 for result in results])),
        metrics=metrics,
    )


def finetune_result_from_scores(
    horizon: int,
    fold: int,
    epoch: int,
    sampling_strategy: str,
    n_context_samples: int,
    y_val: np.ndarray,
    val_score: np.ndarray,
    y_test: np.ndarray,
    test_score: np.ndarray,
) -> TabPFNFinetuneEpochResult:
    threshold = find_best_f1_threshold(y_val, val_score)
    val_metrics = binary_classification_metrics(y_val, val_score, threshold)
    metrics = binary_classification_metrics(y_test, test_score, threshold)
    return TabPFNFinetuneEpochResult(
        model="tabpfn_finetuned",
        horizon=horizon,
        fold=fold,
        epoch=epoch,
        sampling_strategy=sampling_strategy,
        n_context_samples=n_context_samples,
        validation_f1=val_metrics["f1"],
        metrics=metrics,
    )


def finetune_horizon_slice_results_from_scores(
    horizons: list[int],
    fold: int,
    epoch: int,
    sampling_strategy: str,
    n_context_samples: int,
    y_val: np.ndarray,
    val_score: np.ndarray,
    val_horizons: np.ndarray,
    y_test: np.ndarray,
    test_score: np.ndarray,
    test_horizons: np.ndarray,
) -> list[TabPFNFinetuneEpochResult]:
    results = []
    for horizon in horizons:
        val_mask = val_horizons == horizon
        test_mask = test_horizons == horizon
        if not val_mask.any() or not test_mask.any():
            raise ValueError(
                f"Cannot evaluate horizon {horizon}: "
                f"validation rows={int(val_mask.sum())}, "
                f"test rows={int(test_mask.sum())}."
            )
        result = finetune_result_from_scores(
            horizon=horizon,
            fold=fold,
            epoch=epoch,
            sampling_strategy=sampling_strategy,
            n_context_samples=n_context_samples,
            y_val=y_val[val_mask],
            val_score=val_score[val_mask],
            y_test=y_test[test_mask],
            test_score=test_score[test_mask],
        )
        result.metrics["validation_rows"] = int(val_mask.sum())
        result.metrics["test_rows"] = int(test_mask.sum())
        results.append(result)
    return results


def make_finetunable_tabpfn_classifier(
    config: TabPFNFinetuneConfig,
) -> tuple[Any, dict[str, Any]]:
    classifier_config: dict[str, Any] = {
        "ignore_pretraining_limits": True,
        "n_estimators": 1,
        "random_state": config.random_state,
        "inference_precision": resolve_tabpfn_inference_precision(
            config.inference_precision, torch
        ),
    }
    if config.device != "auto":
        classifier_config["device"] = config.device
    if config.model_path is not None:
        classifier_config["model_path"] = config.model_path

    classifier = TabPFNClassifier(
        **classifier_config,
        fit_mode="batched",
        differentiable_input=False,
    )
    classifier._initialize_model_variables()
    classifier.softmax_temperature_ = classifier.softmax_temperature
    if len(classifier.models_) != 1:
        raise ValueError("TabPFN fine-tuning requires n_estimators=1.")
    return classifier, classifier_config


def clone_tabpfn_for_evaluation(classifier, classifier_config: dict[str, Any]):
    eval_config = build_evaluation_clone_config(classifier_config)
    return clone_model_for_evaluation(classifier, eval_config, TabPFNClassifier)


def build_evaluation_clone_config(classifier_config: dict[str, Any]) -> dict[str, Any]:
    eval_config = {
        key: value for key, value in classifier_config.items() if key != "model_path"
    }
    eval_config["inference_config"] = {
        "SUBSAMPLE_SAMPLES": classifier_config.get("SUBSAMPLE_SAMPLES", 10_000)
    }
    return eval_config


def resolve_tabpfn_inference_precision(value: str, torch_module: Any) -> Any:
    normalized = value.lower()
    if normalized in {"auto", "autocast"}:
        return normalized

    dtype_by_name = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float": torch_module.float32,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    if normalized in dtype_by_name:
        return dtype_by_name[normalized]

    raise ValueError(
        "Unsupported TabPFN inference_precision. Use one of: auto, autocast, "
        "float32, float16, bfloat16."
    )


def _predict_positive_proba(
    classifier,
    X: np.ndarray,
    batch_size: int | None,
) -> np.ndarray:
    if batch_size is None or len(X) <= batch_size:
        return classifier.predict_proba(X)[:, 1]
    if batch_size <= 0:
        raise ValueError("eval_batch_size must be positive or null.")

    scores = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        scores.append(classifier.predict_proba(X[start:stop])[:, 1])
    return np.concatenate(scores)


def select_best_epoch(
    results: list[TabPFNFinetuneEpochResult],
) -> TabPFNFinetuneEpochResult:
    if not results:
        raise ValueError("Cannot select best epoch from an empty result list.")
    return max(results, key=lambda result: result.validation_f1)


def _is_better_result(
    result: TabPFNFinetuneEpochResult,
    best: TabPFNFinetuneEpochResult | None,
) -> bool:
    return best is None or result.validation_f1 > best.validation_f1


def select_best_epoch_by_horizon(
    results: list[TabPFNFinetuneEpochResult],
    horizons: list[int] | None = None,
) -> list[TabPFNFinetuneEpochResult]:
    if not results:
        raise ValueError("Cannot select best epochs from an empty result list.")
    selected_horizons = horizons or sorted({result.horizon for result in results})
    best_results = []
    for horizon in selected_horizons:
        horizon_results = [result for result in results if result.horizon == horizon]
        if not horizon_results:
            raise ValueError(f"No results found for horizon {horizon}.")
        best_results.append(select_best_epoch(horizon_results))
    return best_results


def finetune_result_to_row(result: TabPFNFinetuneEpochResult) -> dict[str, Any]:
    row = {
        "model": result.model,
        "horizon": result.horizon,
        "fold": result.fold,
        "epoch": result.epoch,
        "sampling_strategy": result.sampling_strategy,
        "n_context_samples": result.n_context_samples,
        "validation_f1": result.validation_f1,
    }
    row.update(result.metrics)
    return row


def append_finetune_metrics(
    output_dir: str | Path,
    result: TabPFNFinetuneEpochResult,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "metrics.csv"
    pd.DataFrame([finetune_result_to_row(result)]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )
    return path


def append_finetune_horizon_metrics(
    output_dir: str | Path,
    results: list[TabPFNFinetuneEpochResult],
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "metrics_by_horizon.csv"
    pd.DataFrame([finetune_result_to_row(result) for result in results]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )
    return path


def write_best_epoch(
    output_dir: str | Path,
    result: TabPFNFinetuneEpochResult,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "best_epoch.json"
    path.write_text(
        json.dumps(finetune_result_to_row(result), indent=2),
        encoding="utf-8",
    )
    return path


def write_best_epoch_by_horizon(
    output_dir: str | Path,
    results: list[TabPFNFinetuneEpochResult],
) -> tuple[Path, Path]:
    return _write_result_rows(
        output_dir,
        results,
        csv_name="best_epoch_by_horizon.csv",
        json_name="best_epoch_by_horizon.json",
    )


def write_joint_best_epoch_by_horizon(
    output_dir: str | Path,
    results: list[TabPFNFinetuneEpochResult],
) -> tuple[Path, Path]:
    return _write_result_rows(
        output_dir,
        results,
        csv_name="joint_best_epoch_by_horizon.csv",
        json_name="joint_best_epoch_by_horizon.json",
    )


def _write_result_rows(
    output_dir: str | Path,
    results: list[TabPFNFinetuneEpochResult],
    csv_name: str,
    json_name: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = [finetune_result_to_row(result) for result in results]
    csv_path = output_path / csv_name
    json_path = output_path / json_name
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def save_tabpfn_torch_state(output_dir: str | Path, classifier, epoch: int) -> Path:
    return save_tabpfn_torch_state_file(output_dir, classifier, f"epoch_{epoch}.pt")


def save_tabpfn_torch_state_file(
    output_dir: str | Path,
    classifier,
    filename: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename
    torch.save(classifier.models_[0].state_dict(), path)
    return path


def _prepare_arrays(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: TabPFNFinetuneConfig,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx = subsample_indices(
        train_idx,
        config.max_train_samples,
        config.random_state,
    )
    val_idx = subsample_indices(
        val_idx,
        config.max_eval_samples,
        config.random_state + 1,
    )
    test_idx = subsample_indices(
        test_idx,
        config.max_eval_samples,
        config.random_state + 2,
    )
    prepared = split_features_target(df, target_col=target_col)
    return _preprocess_prepared_split_frames(
        train_X=prepared.X.iloc[train_idx],
        train_y=prepared.y.iloc[train_idx].to_numpy(),
        val_X=prepared.X.iloc[val_idx],
        val_y=prepared.y.iloc[val_idx].to_numpy(),
        test_X=prepared.X.iloc[test_idx],
        test_y=prepared.y.iloc[test_idx].to_numpy(),
        config=config,
    )


def _prepare_arrays_from_split_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TabPFNFinetuneConfig,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = _subsample_frame(
        train_df,
        config.max_train_samples,
        config.random_state,
    )
    val_df = _subsample_frame(
        val_df,
        config.max_eval_samples,
        config.random_state + 1,
    )
    test_df = _subsample_frame(
        test_df,
        config.max_eval_samples,
        config.random_state + 2,
    )

    train_prepared = split_features_target(train_df, target_col=target_col)
    val_prepared = split_features_target(val_df, target_col=target_col)
    test_prepared = split_features_target(test_df, target_col=target_col)
    return _preprocess_prepared_split_frames(
        train_X=train_prepared.X,
        train_y=train_prepared.y.to_numpy(),
        val_X=val_prepared.X,
        val_y=val_prepared.y.to_numpy(),
        test_X=test_prepared.X,
        test_y=test_prepared.y.to_numpy(),
        config=config,
    )


def _prepare_arrays_and_horizon_labels_from_split_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons: list[int],
    config: TabPFNFinetuneConfig,
    target_col: str,
    horizon_col: str,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    train_df = _subsample_frame(
        train_df,
        config.max_train_samples,
        config.random_state,
    )
    val_df = _subsample_frame(
        val_df,
        config.max_eval_samples,
        config.random_state + 1,
    )
    test_df = _subsample_frame(
        test_df,
        config.max_eval_samples,
        config.random_state + 2,
    )
    train_horizons = _extract_horizon_labels(train_df, horizon_col)
    val_horizons = _extract_horizon_labels(val_df, horizon_col)
    test_horizons = _extract_horizon_labels(test_df, horizon_col)

    train_prepared = split_features_target(train_df, target_col=target_col)
    val_prepared = split_features_target(val_df, target_col=target_col)
    test_prepared = split_features_target(test_df, target_col=target_col)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = _preprocess_prepared_split_frames_without_sampling(
        train_X=train_prepared.X,
        train_y=train_prepared.y.to_numpy(),
        val_X=val_prepared.X,
        val_y=val_prepared.y.to_numpy(),
        test_X=test_prepared.X,
        test_y=test_prepared.y.to_numpy(),
    )
    X_train, y_train, train_horizons = _apply_sampling_by_horizon(
        X_train,
        y_train,
        train_horizons,
        horizons,
        config,
    )
    return (
        (X_train, y_train, X_val, y_val, X_test, y_test),
        train_horizons,
        val_horizons,
        test_horizons,
    )


def _extract_horizon_labels(df: pd.DataFrame, horizon_col: str) -> np.ndarray:
    if horizon_col not in df.columns:
        raise ValueError(f"Missing horizon column: {horizon_col}")
    return df[horizon_col].to_numpy(dtype=int)


def _apply_sampling_by_horizon(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_horizons: np.ndarray,
    horizons: list[int],
    config: TabPFNFinetuneConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_parts = []
    y_parts = []
    horizon_parts = []
    for horizon in horizons:
        mask = train_horizons == horizon
        if not mask.any():
            raise ValueError(f"Cannot sample horizon {horizon}: no train rows.")
        X_horizon, y_horizon = apply_sampling(
            X_train[mask],
            y_train[mask],
            SamplingConfig(
                strategy=config.sampling_strategy,
                random_state=config.random_state + horizon,
                minority_to_majority_ratio=config.minority_to_majority_ratio,
                prototype_backend=config.prototype_backend,
            ),
        )
        X_parts.append(X_horizon)
        y_parts.append(y_horizon)
        horizon_parts.append(np.full(len(y_horizon), horizon, dtype=int))

    X_sampled = np.vstack(X_parts)
    y_sampled = np.hstack(y_parts)
    horizons_sampled = np.hstack(horizon_parts)
    order = np.random.default_rng(config.random_state).permutation(len(y_sampled))
    return X_sampled[order], y_sampled[order], horizons_sampled[order]


def _preprocess_prepared_split_frames(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_X: pd.DataFrame,
    val_y: np.ndarray,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
    config: TabPFNFinetuneConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        X_train_np,
        y_train,
        X_val_np,
        y_val_np,
        X_test_np,
        y_test_np,
    ) = _preprocess_prepared_split_frames_without_sampling(
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
    )
    X_train_np, y_train = apply_sampling(
        X_train_np,
        y_train,
        SamplingConfig(
            strategy=config.sampling_strategy,
            random_state=config.random_state,
            minority_to_majority_ratio=config.minority_to_majority_ratio,
            prototype_backend=config.prototype_backend,
        ),
    )
    return (
        X_train_np,
        y_train,
        X_val_np,
        y_val_np,
        X_test_np,
        y_test_np,
    )


def _preprocess_prepared_split_frames_without_sampling(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_X: pd.DataFrame,
    val_y: np.ndarray,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_val, X_test, _ = preprocess_train_val_test(
        train_X,
        val_X,
        test_X,
    )
    X_train_np = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=int)
    return (
        X_train_np,
        y_train,
        np.asarray(X_val, dtype=np.float64),
        np.asarray(val_y, dtype=int),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(test_y, dtype=int),
    )


def _subsample_frame(
    df: pd.DataFrame,
    max_samples: int | None,
    random_state: int,
) -> pd.DataFrame:
    indices = subsample_indices(np.arange(len(df)), max_samples, random_state)
    return df.iloc[indices]


def _make_optimizer(classifier, config: TabPFNFinetuneConfig):
    try:
        from torch.optim import Adam
    except ImportError as exc:
        raise ImportError("TabPFN fine-tuning requires Torch.") from exc
    return Adam(classifier.models_[0].parameters(), lr=config.learning_rate)


def _make_finetuning_dataloader(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TabPFNFinetuneConfig,
):
    try:
        from tabpfn.utils import meta_dataset_collator
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "TabPFN fine-tuning dataloaders require TabPFN and Torch."
        ) from exc

    splitter = lambda X, y: train_test_split(  # noqa: E731
        X,
        y,
        test_size=0.3,
        random_state=config.random_state,
        stratify=y,
    )
    training_datasets = classifier.get_preprocessed_datasets(
        X_train,
        y_train,
        splitter,
        config.batch_size,
    )
    return DataLoader(
        training_datasets,
        batch_size=config.meta_batch_size,
        collate_fn=meta_dataset_collator,
    )


def _make_horizon_conditioned_finetuning_dataloaders(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_horizons: np.ndarray,
    horizons: list[int],
    config: TabPFNFinetuneConfig,
):
    return [
        _make_finetuning_dataloader(
            classifier,
            X_train[train_horizons == horizon],
            y_train[train_horizons == horizon],
            config,
        )
        for horizon in horizons
    ]


def _finetune_one_epoch(classifier, optimizer, dataloader, config, epoch: int) -> None:
    loss_function = _build_loss_function(config)
    progress = tqdm(dataloader, desc=f"TabPFN finetune epoch {epoch}", unit="batch")
    for batch in progress:
        loss = _finetune_one_batch(classifier, optimizer, batch, config, loss_function)
        if loss is not None:
            progress.set_postfix(loss=f"{loss:.4f}")


def _finetune_one_horizon_conditioned_epoch(
    classifier,
    optimizer,
    dataloaders,
    config: TabPFNFinetuneConfig,
    epoch: int,
) -> None:
    loss_function = _build_loss_function(config)
    iterators = [iter(dataloader) for dataloader in dataloaders]
    active = list(range(len(iterators)))
    rng = np.random.default_rng(config.random_state + epoch)
    total = sum(len(dataloader) for dataloader in dataloaders)
    progress = tqdm(
        total=total,
        desc=f"TabPFN horizon-conditioned finetune epoch {epoch}",
        unit="batch",
    )
    while active:
        for loader_index in rng.permutation(active).tolist():
            if loader_index not in active:
                continue
            try:
                batch = next(iterators[loader_index])
            except StopIteration:
                active.remove(loader_index)
                continue
            loss = _finetune_one_batch(
                classifier,
                optimizer,
                batch,
                config,
                loss_function,
            )
            progress.update(1)
            if loss is not None:
                progress.set_postfix(loss=f"{loss:.4f}")
    progress.close()


def _finetune_one_batch(
    classifier,
    optimizer,
    batch,
    config: TabPFNFinetuneConfig,
    loss_function,
) -> float | None:
    X_train_batch, X_test_batch, y_train_batch, y_test_batch, cat_ixs, confs = batch
    if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
        return None
    optimizer.zero_grad()
    classifier.fit_from_preprocessed(X_train_batch, y_train_batch, cat_ixs, confs)
    predictions = classifier.forward(X_test_batch, return_logits=True)
    if config.loss_function == "sigmoid_focal_loss":
        loss = _sigmoid_focal_loss(predictions, y_test_batch, config)
    else:
        target = y_test_batch.to(_batch_device(predictions, config))
        loss = loss_function(predictions, target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _build_loss_function(config: TabPFNFinetuneConfig):
    if config.loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    if config.loss_function == "sigmoid_focal_loss":
        return None
    raise ValueError(f"Unknown loss function: {config.loss_function}")


def _sigmoid_focal_loss(predictions, y_test_batch, config: TabPFNFinetuneConfig):
    logits_positive = predictions[:, 1] - predictions[:, 0]
    return sigmoid_focal_loss(
        inputs=logits_positive.unsqueeze(1),
        targets=y_test_batch.float()
        .unsqueeze(1)
        .to(_batch_device(predictions, config)),
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
        reduction="mean",
    )


def _batch_device(predictions, config: TabPFNFinetuneConfig):
    return predictions.device if config.device == "auto" else config.device


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
