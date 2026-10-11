import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from v4finbench.data.preprocessing import prepare_split
from v4finbench.data.splits import TemporalSplit
from v4finbench.evaluation.metrics import binary_classification_metrics
from v4finbench.evaluation.protocol import evaluate_test_scores, unseen_firm_mask
from v4finbench.evaluation.results import (
    RESULT_KEY,
    RunResult,
    upsert_rows,
    write_test_scores,
)
from v4finbench.evaluation.thresholds import find_best_f1_threshold
from v4finbench.models.tabpfn.sampling import SamplingConfig, apply_sampling
from v4finbench.models.tabpfn.vanilla import (
    predict_proba_batched,
    select_context_samples,
    subsample_split,
)


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
    split: TemporalSplit,
    config: TabPFNFinetuneConfig | None = None,
    output_dir: str | Path | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> list[RunResult]:
    config = config or TabPFNFinetuneConfig()
    train_idx, val_idx, test_idx = subsample_split(
        split, config.max_train_samples, config.max_eval_samples, config.random_state
    )
    data = prepare_split(df, train_idx, val_idx, test_idx)
    X_train, y_train = apply_sampling(
        np.asarray(data.X_train, dtype=np.float64),
        data.y_train,
        SamplingConfig(
            strategy=config.sampling_strategy,
            random_state=config.random_state,
            minority_to_majority_ratio=config.minority_to_majority_ratio,
        ),
    )
    X_val = np.asarray(data.X_val, dtype=np.float64)
    X_test = np.asarray(data.X_test, dtype=np.float64)
    unseen = unseen_firm_mask(df, train_idx, test_idx)

    classifier, classifier_config = (
        classifier_factory(config)
        if classifier_factory is not None
        else make_finetunable_tabpfn_classifier(config)
    )
    optimizer = _make_optimizer(classifier, config)
    dataloader = _make_finetuning_dataloader(classifier, X_train, y_train, config)

    results = []
    for epoch in range(config.epochs + 1):
        epoch_start = time.perf_counter()
        if epoch > 0:
            _finetune_one_epoch(classifier, optimizer, dataloader, config, epoch)
        train_seconds = time.perf_counter() - epoch_start

        eval_classifier = clone_tabpfn_for_evaluation(classifier, classifier_config)
        X_context, y_context = select_context_samples(
            X_train,
            y_train,
            n_context_samples=config.n_inference_context_samples,
            random_state=config.random_state,
        )
        eval_classifier.fit(X_context, y_context)
        val_score = predict_proba_batched(eval_classifier, X_val)
        threshold = find_best_f1_threshold(data.y_val, val_score)
        validation_f1 = binary_classification_metrics(
            data.y_val, val_score, threshold
        )["f1"]
        test_score = predict_proba_batched(eval_classifier, X_test)

        result = RunResult(
            model=f"tabpfn_finetuned_{config.sampling_strategy}",
            horizon=split.horizon,
            test_year=split.test_year,
            validation_f1=validation_f1,
            metrics=evaluate_test_scores(data.y_test, test_score, threshold, unseen),
            details={
                "epoch": epoch,
                "sampling_strategy": config.sampling_strategy,
                "n_context_samples": len(y_context),
                "train_seconds": round(train_seconds, 2),
            },
            test_idx=test_idx,
            test_labels=data.y_test,
            test_scores=test_score,
            unseen_firm=unseen,
        )
        results.append(result)
        if output_dir is not None:
            upsert_rows(
                Path(output_dir) / "metrics.csv",
                [result.to_row()],
                key=[*RESULT_KEY, "epoch"],
            )
            if config.save_checkpoints:
                save_tabpfn_torch_state(output_dir, classifier, epoch)

    if output_dir is not None:
        best = select_best_epoch(results)
        (Path(output_dir) / "best_epoch.json").write_text(
            json.dumps(best.to_row(), indent=2), encoding="utf-8"
        )
        write_test_scores(Path(output_dir) / "test_scores.parquet", best, df)
    return results


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
        # bfloat16 breaks torch.unique in TabPFN's y-encoder on CUDA.
        "inference_precision": torch.float32,
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
        "inference_precision": torch.float32,
    }
    # clone_model_for_evaluation passes model_path itself.
    eval_config.pop("model_path", None)
    return clone_model_for_evaluation(classifier, eval_config, TabPFNClassifier)


def select_best_epoch(results: list[RunResult]) -> RunResult:
    if not results:
        raise ValueError("Cannot select best epoch from an empty result list.")
    return max(results, key=lambda result: result.validation_f1)


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
