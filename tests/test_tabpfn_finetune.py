import numpy as np

from v4finbench.models.tabpfn_finetune import (
    TabPFNFinetuneEpochResult,
    _predict_positive_proba,
    build_evaluation_clone_config,
    finetune_config_from_mapping,
    finetune_result_to_row,
    resolve_tabpfn_inference_precision,
    select_best_epoch,
)


def test_finetune_config_from_mapping_reads_nested_finetuning() -> None:
    config = finetune_config_from_mapping(
        {
            "sampling_strategy": "prototype_undersampling",
            "random_seed": 768,
            "model_path": "weights/tabpfn.ckpt",
            "n_inference_context_samples": 10000,
            "eval_batch_size": 2048,
            "device": "cuda",
            "inference_precision": "auto",
            "prototype_backend": "cuml",
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
    assert config.eval_batch_size == 2048
    assert config.device == "cuda"
    assert config.inference_precision == "auto"
    assert config.prototype_backend == "cuml"
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


def test_build_evaluation_clone_config_drops_model_path() -> None:
    config = build_evaluation_clone_config(
        {
            "model_path": "weights/tabpfn.ckpt",
            "device": "cuda",
            "random_state": 42,
        }
    )

    assert "model_path" not in config
    assert config["device"] == "cuda"
    assert config["inference_config"] == {"SUBSAMPLE_SAMPLES": 10_000}


def test_resolve_tabpfn_inference_precision_keeps_auto_modes() -> None:
    class TorchStub:
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"

    assert resolve_tabpfn_inference_precision("auto", TorchStub) == "auto"
    assert resolve_tabpfn_inference_precision("autocast", TorchStub) == "autocast"
    assert resolve_tabpfn_inference_precision("bfloat16", TorchStub) == "bfloat16"


def test_predict_positive_proba_batches_large_eval_arrays() -> None:
    class Classifier:
        def __init__(self) -> None:
            self.call_sizes = []

        def predict_proba(self, X):
            self.call_sizes.append(len(X))
            return np.asarray([[1 - row[0], row[0]] for row in X])

    classifier = Classifier()

    scores = _predict_positive_proba(
        classifier,
        [[0.1], [0.2], [0.3], [0.4], [0.5]],
        batch_size=2,
    )

    assert classifier.call_sizes == [2, 2, 1]
    assert scores.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]


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
