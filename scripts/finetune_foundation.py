import argparse
import json
from pathlib import Path

import yaml

from v4finbench.data.io import iter_splits
from v4finbench.data.schema import HORIZONS
from v4finbench.data.splits import TRACKS
from v4finbench.evaluation.results import hardware, write_test_scores
from v4finbench.models.foundation.finetune import (
    SAMPLING_NAMES,
    finetuned_model_name,
    run_finetune,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune TabPFN or TabICL on the walk-forward tasks."
    )
    parser.add_argument("--estimator", choices=["tabpfn", "tabicl"], required=True)
    parser.add_argument("--sampling", choices=list(SAMPLING_NAMES), required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/foundation/finetune.yaml")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results/generated/finetune"))
    parser.add_argument(
        "--horizons", type=int, nargs="+", choices=HORIZONS, default=HORIZONS
    )
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--track", choices=TRACKS, default="main")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    model_name = finetuned_model_name(args.estimator, args.sampling)

    for df, split in iter_splits(
        args.data_dir, args.horizons, args.test_year, args.track
    ):
        cell = args.out / f"h{split.horizon}" / f"Y{split.test_year}"
        metrics_path = cell / f"{model_name}.metrics.json"
        if metrics_path.exists() and not args.overwrite:
            print(f"Skipping {metrics_path} (exists)", flush=True)
            continue
        result = run_finetune(
            df,
            split,
            args.estimator,
            args.sampling,
            config,
            output_dir=cell / f"{model_name}_weights",
            random_state=args.random_state,
        )
        write_test_scores(cell / f"{model_name}.parquet", result, df)
        metrics_path.write_text(
            json.dumps(result.to_row() | hardware(), indent=2, default=str),
            encoding="utf-8",
        )
        m = result.metrics
        print(
            f"{model_name} h{split.horizon} Y{split.test_year}: "
            f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
            f"lr={result.details['learning_rate']:g} "
            f"epoch={result.details['best_epoch']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
