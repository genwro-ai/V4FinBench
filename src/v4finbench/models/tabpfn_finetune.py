import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from v4finbench.data.preprocessing import (
    preprocess_train_val_test,
    split_features_target,
)
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.thresholds import find_best_f1_threshold
from v4finbench.models.tabpfn import select_context_samples, subsample_indices
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
    model_path: str | None = None
    epochs: int = 10
    learning_rate: float = 5e-6
    batch_size: int = 1024
    meta_batch_size: int = 1
    loss_function: str = "cross_entropy"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    save_checkpoints: bool = False


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
    # Per-instance test probabilities for this epoch; kept out of the metrics
    # CSV and written to test_scores.parquet for the best epoch only.
    test_scores: np.ndarray | None = None


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
        model_path=values.get("model_path"),
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
    X_train, y_train, X_val, y_val, X_test, y_test = arrays

    classifier, classifier_config = (
        classifier_factory(config)
        if classifier_factory is not None
        else make_finetunable_tabpfn_classifier(config)
    )
    optimizer = _make_optimizer(classifier, config)
    dataloader = _make_finetuning_dataloader(classifier, X_train, y_train, config)

    results = []
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
            if config.save_checkpoints:
                save_tabpfn_torch_state(output_dir, classifier, epoch)

    if output_dir is not None:
        best = select_best_epoch(results)
        write_best_epoch(output_dir, best)
        if best.test_scores is not None:
            # Reproduce the deterministic test subsample from _prepare_arrays
            # so row_idx maps back to rows of the source parquet file.
            effective_test_idx = subsample_indices(
                test_idx,
                config.max_eval_samples,
                config.random_state + 2,
            )
            pd.DataFrame(
                {
                    "row_idx": np.asarray(effective_test_idx, dtype=np.int64),
                    "y_true": np.asarray(y_test, dtype=np.int64),
                    "score": best.test_scores,
                }
            ).to_parquet(Path(output_dir) / "test_scores.parquet", index=False)
    return results


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

    val_score = eval_classifier.predict_proba(X_val)[:, 1]
    threshold = find_best_f1_threshold(y_val, val_score)
    val_metrics = binary_classification_metrics(y_val, val_score, threshold)

    test_score = eval_classifier.predict_proba(X_test)[:, 1]
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
        test_scores=test_score,
    )


def make_finetunable_tabpfn_classifier(
    config: TabPFNFinetuneConfig,
) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise ImportError(
            "TabPFN fine-tuning requires TabPFN and Torch. "
            "Run `uv sync --extra tabpfn` first."
        ) from exc

    classifier_config: dict[str, Any] = {
        "ignore_pretraining_limits": True,
        "n_estimators": 1,
        "random_state": config.random_state,
        "inference_precision": torch.bfloat16,
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
    try:
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.finetune_utils import clone_model_for_evaluation
    except ImportError as exc:
        raise ImportError(
            "TabPFN evaluation cloning requires `uv sync --extra tabpfn`."
        ) from exc

    eval_config = {
        **classifier_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": classifier_config.get("SUBSAMPLE_SAMPLES", 10_000)
        },
        # Evaluation must run in float32: with bfloat16 inference precision,
        # TabPFN's label encoder calls torch.unique on a BFloat16 tensor on
        # the CUDA path, and torch has no BFloat16 kernel for unique
        # ('"unique" not implemented for BFloat16'). Finetuning keeps
        # bfloat16, where the throughput matters.
        "inference_precision": torch.float32,
    }
    # clone_model_for_evaluation derives the model spec from the fitted
    # classifier and passes model_path itself. Leaving the key in the config
    # raises "got multiple values for keyword argument 'model_path'" (only
    # visible when a --model-path is provided, hence not caught by the
    # model_path-less smoke test).
    eval_config.pop("model_path", None)
    return clone_model_for_evaluation(classifier, eval_config, TabPFNClassifier)


def select_best_epoch(
    results: list[TabPFNFinetuneEpochResult],
) -> TabPFNFinetuneEpochResult:
    if not results:
        raise ValueError("Cannot select best epoch from an empty result list.")
    return max(results, key=lambda result: result.validation_f1)


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


def save_tabpfn_torch_state(output_dir: str | Path, classifier, epoch: int) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Saving TabPFN checkpoints requires Torch.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"epoch_{epoch}.pt"
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
    X_train_raw = prepared.X.iloc[train_idx]
    X_val_raw = prepared.X.iloc[val_idx]
    X_test_raw = prepared.X.iloc[test_idx]
    y_train = prepared.y.iloc[train_idx].to_numpy()
    y_val = prepared.y.iloc[val_idx].to_numpy()
    y_test = prepared.y.iloc[test_idx].to_numpy()

    X_train, X_val, X_test, _ = preprocess_train_val_test(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )
    X_train_np = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    X_train_np, y_train = apply_sampling(
        X_train_np,
        y_train,
        SamplingConfig(
            strategy=config.sampling_strategy,
            random_state=config.random_state,
            minority_to_majority_ratio=config.minority_to_majority_ratio,
        ),
    )
    return (
        X_train_np,
        y_train,
        np.asarray(X_val, dtype=np.float64),
        np.asarray(y_val, dtype=int),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=int),
    )


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


def _finetune_one_epoch(classifier, optimizer, dataloader, config, epoch: int) -> None:
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("TabPFN fine-tuning requires Torch and tqdm.") from exc

    loss_function = _build_loss_function(config)
    progress = tqdm(dataloader, desc=f"TabPFN finetune epoch {epoch}", unit="batch")
    for batch in progress:
        X_train_batch, X_test_batch, y_train_batch, y_test_batch, cat_ixs, confs = batch
        if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
            continue
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
        progress.set_postfix(loss=f"{float(loss.item()):.4f}")


def _build_loss_function(config: TabPFNFinetuneConfig):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("TabPFN fine-tuning requires Torch.") from exc
    if config.loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    if config.loss_function == "sigmoid_focal_loss":
        return None
    raise ValueError(f"Unknown loss function: {config.loss_function}")


def _sigmoid_focal_loss(predictions, y_test_batch, config: TabPFNFinetuneConfig):
    try:
        from torchvision.ops import sigmoid_focal_loss
    except ImportError as exc:
        raise ImportError("Focal loss requires torchvision.") from exc

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
