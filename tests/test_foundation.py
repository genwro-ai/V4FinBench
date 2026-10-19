import pytest
import torch

from v4finbench.models.baselines import run_baseline
from v4finbench.models.foundation.finetune import best_checkpoint, finetuned_model_name


def test_finetuned_model_names() -> None:
    assert finetuned_model_name("tabpfn", "none") == "tabpfn_finetuned_full"
    assert (
        finetuned_model_name("tabicl", "prototype_undersampling")
        == "tabicl_finetuned_proto"
    )


def test_best_checkpoint_falls_back_to_pretrained(tmp_path) -> None:
    assert best_checkpoint(tmp_path) == (0, "pretrained")
    torch.save({"epoch": 3}, tmp_path / "best.ckpt")
    assert best_checkpoint(tmp_path) == (3, str(tmp_path / "best.ckpt"))


@pytest.mark.parametrize("sampling", ["none", "undersample"])
def test_in_context_tabpfn(torch_module, labeled_frame, split_2012, sampling) -> None:
    config = {
        "n_trials": 1,
        "models": {
            "m": {
                "estimator": "tabpfn",
                "params": {"sampling": sampling, "ratio": 0.5, "device": "cpu"},
            }
        },
    }
    result = run_baseline(labeled_frame, split_2012, "m", config)

    assert result.metrics["roc_auc"] > 0.9
    assert len(result.test_scores) == len(split_2012.test_idx)
