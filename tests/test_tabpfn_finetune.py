from v4finbench.evaluation.results import RunResult
from v4finbench.models.tabpfn.finetune import (
    finetune_config_from_mapping,
    select_best_epoch,
)


def test_finetune_config_from_mapping_reads_nested_finetuning() -> None:
    config = finetune_config_from_mapping(
        {
            "random_seed": 768,
            "n_inference_context_samples": 10000,
            "finetuning": {"epochs": 3, "learning_rate": 1e-5, "batch_size": 512},
        }
    )

    assert config.random_state == 768
    assert config.epochs == 3
    assert config.learning_rate == 1e-5
    assert config.batch_size == 512


def test_select_best_epoch_uses_validation_f1() -> None:
    def epoch(number: int, validation_f1: float) -> RunResult:
        return RunResult(
            model="tabpfn_finetuned",
            horizon=1,
            test_year=2015,
            validation_f1=validation_f1,
            metrics={"f1": 0.0},
            details={"epoch": number},
        )

    assert select_best_epoch([epoch(0, 0.1), epoch(1, 0.4)]).details["epoch"] == 1
