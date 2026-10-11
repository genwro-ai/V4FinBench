import argparse
from pathlib import Path

import yaml

from v4finbench.data.io import read_horizon
from v4finbench.data.splits import get_split
from v4finbench.models.tabpfn.finetune import (
    finetune_config_from_mapping,
    finetune_evaluate_tabpfn,
    select_best_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune TabPFN on one walk-forward split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tabpfn/finetune_prototype_undersample.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/tabpfn_finetune")
    )
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    overrides = {
        "model_path": args.model_path,
        "device": args.device,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = finetune_config_from_mapping(values)

    df = read_horizon(args.data_dir, args.horizon)
    split = get_split(df, args.horizon, args.test_year)
    output_dir = args.out / f"h{args.horizon}" / f"Y{args.test_year}"
    for stale in ("metrics.csv", "best_epoch.json", "test_scores.parquet"):
        (output_dir / stale).unlink(missing_ok=True)

    results = finetune_evaluate_tabpfn(df, split, config, output_dir=output_dir)
    best = select_best_epoch(results)
    for checkpoint in output_dir.glob("epoch_*.pt"):
        if checkpoint.stem != f"epoch_{best.details['epoch']}":
            checkpoint.unlink()
    print(
        f"Best epoch={best.details['epoch']}: "
        f"validation_f1={best.validation_f1:.4f}, test_f1={best.metrics['f1']:.4f}, "
        f"test_roc_auc={best.metrics['roc_auc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
