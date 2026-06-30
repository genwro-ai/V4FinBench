import numpy as np

from v4finbench.models.tabpfn_finetune import (
    TabPFNFinetuneConfig,
    TabPFNFinetuneEpochResult,
    _apply_sampling_by_horizon,
    _predict_positive_proba,
    build_evaluation_clone_config,
    finetune_config_from_mapping,
    finetune_horizon_slice_results_from_scores,
    finetune_result_to_row,
    macro_average_finetune_results,
    resolve_tabpfn_inference_precision,
    select_best_epoch,
    select_best_epoch_by_horizon,
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
            "save_best_checkpoint": False,
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
    assert config.save_best_checkpoint is False
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


def test_select_best_epoch_by_horizon_uses_each_horizon_validation_f1() -> None:
    results = [
        TabPFNFinetuneEpochResult(
            model="tabpfn_finetuned",
            horizon=0,
            fold=0,
            epoch=0,
            sampling_strategy="none",
            n_context_samples=10,
            validation_f1=0.1,
            metrics={"f1": 0.2},
        ),
        TabPFNFinetuneEpochResult(
            model="tabpfn_finetuned",
            horizon=0,
            fold=0,
            epoch=1,
            sampling_strategy="none",
            n_context_samples=10,
            validation_f1=0.4,
            metrics={"f1": 0.3},
        ),
        TabPFNFinetuneEpochResult(
            model="tabpfn_finetuned",
            horizon=1,
            fold=0,
            epoch=0,
            sampling_strategy="none",
            n_context_samples=10,
            validation_f1=0.5,
            metrics={"f1": 0.6},
        ),
        TabPFNFinetuneEpochResult(
            model="tabpfn_finetuned",
            horizon=1,
            fold=0,
            epoch=1,
            sampling_strategy="none",
            n_context_samples=10,
            validation_f1=0.2,
            metrics={"f1": 0.1},
        ),
    ]

    best = select_best_epoch_by_horizon(results, horizons=[0, 1])

    assert [result.horizon for result in best] == [0, 1]
    assert [result.epoch for result in best] == [1, 0]


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


def test_finetune_horizon_slice_results_use_horizon_thresholds() -> None:
    results = finetune_horizon_slice_results_from_scores(
        horizons=[0, 1],
        fold=2,
        epoch=3,
        sampling_strategy="prototype_undersampling",
        n_context_samples=100,
        y_val=np.asarray([0, 1, 0, 1]),
        val_score=np.asarray([0.2, 0.8, 0.4, 0.6]),
        val_horizons=np.asarray([0, 0, 1, 1]),
        y_test=np.asarray([0, 1, 0, 1]),
        test_score=np.asarray([0.1, 0.9, 0.3, 0.7]),
        test_horizons=np.asarray([0, 0, 1, 1]),
    )

    assert [result.horizon for result in results] == [0, 1]
    assert [result.validation_f1 for result in results] == [1.0, 1.0]
    assert [result.metrics["threshold"] for result in results] == [0.8, 0.6]
    assert [result.metrics["f1"] for result in results] == [1.0, 1.0]
    assert [result.metrics["validation_rows"] for result in results] == [2, 2]
    assert [result.metrics["test_rows"] for result in results] == [2, 2]


def test_horizon_conditioned_sampling_preserves_horizon_labels() -> None:
    X = np.asarray(
        [
            [0.0, 0.1],
            [0.0, 0.2],
            [1.0, 1.1],
            [1.0, 1.2],
        ]
    )
    y = np.asarray([0, 1, 0, 1])
    horizons = np.asarray([0, 0, 1, 1])

    X_sampled, y_sampled, sampled_horizons = _apply_sampling_by_horizon(
        X,
        y,
        horizons,
        [0, 1],
        TabPFNFinetuneConfig(sampling_strategy="none"),
    )

    assert len(y_sampled) == 4
    assert set(sampled_horizons.tolist()) == {0, 1}
    assert np.all(X_sampled[:, 0] == sampled_horizons)


def test_macro_average_finetune_results_uses_mean_validation_f1() -> None:
    results = [
        TabPFNFinetuneEpochResult(
            model="tabpfn_finetuned",
            horizon=0,
            fold=0,
            epoch=1,
            sampling_strategy="prototype_undersampling",
            n_context_samples=5,
            validation_f1=0.2,
            metrics={"f1": 0.3, "validation_rows": 10, "test_rows": 20},
        ),
        TabPFNFinetuneEpochResult(
            model="tabpfn_finetuned",
            horizon=1,
            fold=0,
            epoch=1,
            sampling_strategy="prototype_undersampling",
            n_context_samples=7,
            validation_f1=0.6,
            metrics={"f1": 0.7, "validation_rows": 11, "test_rows": 21},
        ),
    ]

    macro = macro_average_finetune_results(
        results,
        horizon=-1,
        fold=0,
        epoch=1,
        sampling_strategy="prototype_undersampling",
    )

    assert macro.horizon == -1
    assert macro.validation_f1 == 0.4
    assert macro.metrics["f1"] == 0.5
    assert macro.metrics["validation_rows"] == 21
    assert macro.metrics["test_rows"] == 41
    assert macro.n_context_samples == 12
