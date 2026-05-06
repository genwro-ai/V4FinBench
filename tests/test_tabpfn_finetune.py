from __future__ import annotations

from v4finbench.models.tabpfn_finetune import (
    TabPFNFinetuneEpochResult,
    finetune_config_from_mapping,
    finetune_result_to_row,
    select_best_epoch,
)


def test_finetune_config_from_mapping_reads_nested_finetuning() -> None:
    config = finetune_config_from_mapping(
        {
            "sampling_strategy": "prototype_undersampling",
            "random_seed": 768,
            "model_path": "weights/tabpfn.ckpt",
            "n_inference_context_samples": 10000,
            "finetuning": {
                "epochs": 3,
                "learning_rate": 1e-5,
                "batch_size": 512,
                "meta_batch_size": 2,
            },
        }
    )

    assert config.sampling_strategy == "prototype_undersampling"
    assert config.random_state == 768
    assert config.model_path == "weights/tabpfn.ckpt"
    assert config.epochs == 3
    assert config.learning_rate == 1e-5
    assert config.batch_size == 512
    assert config.meta_batch_size == 2


def test_select_best_epoch_uses_validation_f1() -> None:
    low = TabPFNFinetuneEpochResult(
        model="tabpfn_finetuned",
        horizon=0,
        fold=0,
        epoch=0,
        sampling_strategy="none",
        n_context_samples=10,
        validation_f1=0.1,
        metrics={"f1": 0.2},
    )
    high = TabPFNFinetuneEpochResult(
        model="tabpfn_finetuned",
        horizon=0,
        fold=0,
        epoch=1,
        sampling_strategy="none",
        n_context_samples=10,
        validation_f1=0.4,
        metrics={"f1": 0.3},
    )

    assert select_best_epoch([low, high]) == high


def test_finetune_result_to_row_flattens_metrics() -> None:
    result = TabPFNFinetuneEpochResult(
        model="tabpfn_finetuned",
        horizon=2,
        fold=1,
        epoch=3,
        sampling_strategy="prototype_undersampling",
        n_context_samples=100,
        validation_f1=0.4,
        metrics={"f1": 0.3, "roc_auc": 0.9},
    )

    row = finetune_result_to_row(result)

    assert row["epoch"] == 3
    assert row["f1"] == 0.3
    assert row["roc_auc"] == 0.9

