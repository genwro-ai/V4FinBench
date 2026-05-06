from v4finbench.models.tabpfn import tabpfn_config_from_mapping


def test_tabpfn_config_from_mapping_uses_values() -> None:
    config = tabpfn_config_from_mapping(
        {
            "sampling_strategy": "prototype_undersampling",
            "random_seed": 12,
            "n_context_samples": 256,
            "max_train_samples": 1000,
            "max_eval_samples": 500,
            "model_path": "weights/tabpfn.ckpt",
        }
    )

    assert config.sampling_strategy == "prototype_undersampling"
    assert config.n_context_samples == 256
    assert config.max_train_samples == 1000
    assert config.max_eval_samples == 500
    assert config.random_state == 12
    assert config.model_path == "weights/tabpfn.ckpt"
