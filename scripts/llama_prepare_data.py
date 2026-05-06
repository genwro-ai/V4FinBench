import argparse
from pathlib import Path

import pandas as pd
import yaml

from v4finbench.data.schema import HORIZON_FILES
from v4finbench.llama.formatting import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    dataframe_to_llama_records,
    render_system_prompt,
)
from v4finbench.llama.sampling import LlamaSplitConfig, create_llama_train_test_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sampled chat CSVs for the separate Llama QLoRA experiment."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/llama"))
    parser.add_argument("--horizon", default="all", help="'all' or paper horizon 0..5")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--min-positive-ratio-train", type=float, default=None)
    parser.add_argument("--min-positive-ratio-test", type=float, default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = _load_config(args.config)
    values.update(
        {
            key: value
            for key, value in {
                "target_col": args.target_col,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "min_positive_ratio_train": args.min_positive_ratio_train,
                "min_positive_ratio_test": args.min_positive_ratio_test,
                "system_prompt": args.system_prompt,
                "seed": args.seed,
            }.items()
            if value is not None
        }
    )
    if args.system_prompt_file is not None:
        values["system_prompt"] = args.system_prompt_file.read_text(encoding="utf-8")
    config = LlamaSplitConfig(
        train_size=int(values.get("train_size", 20_000)),
        test_size=int(values.get("test_size", 100_000)),
        min_positive_ratio_train=float(values.get("min_positive_ratio_train", 0.10)),
        min_positive_ratio_test=float(values.get("min_positive_ratio_test", 0.10)),
        seed=int(values.get("seed", 42)),
        target_col=str(values.get("target_col", "main_label")),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for horizon in _parse_selection(args.horizon, sorted(HORIZON_FILES)):
        system_prompt = render_system_prompt(
            str(values.get("system_prompt", DEFAULT_SYSTEM_PROMPT_TEMPLATE)),
            horizon=horizon,
        )
        data_path = args.data_dir / HORIZON_FILES[horizon]
        if not data_path.exists():
            raise FileNotFoundError(f"Missing horizon data file: {data_path}")
        print(f"Loading h={horizon}: {data_path}", flush=True)
        df = pd.read_parquet(data_path)
        train_df, test_df = create_llama_train_test_splits(df, config)

        train_records = dataframe_to_llama_records(
            train_df,
            target_col=config.target_col,
            include_label=True,
            system_prompt=system_prompt,
        )
        test_records = dataframe_to_llama_records(
            test_df,
            target_col=config.target_col,
            include_label=True,
            system_prompt=system_prompt,
        )

        train_out = args.out / f"llama_h{horizon}_train.csv"
        test_out = args.out / f"llama_h{horizon}_test.csv"
        train_records.to_csv(train_out, index=False)
        test_records.to_csv(test_out, index=False)
        print(
            f"Wrote h={horizon}: train={train_out} ({len(train_records):,}), "
            f"test={test_out} ({len(test_records):,})",
            flush=True,
        )


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return dict(config)


def _parse_selection(value: str, allowed: list[int]) -> list[int]:
    if value == "all":
        return allowed
    selected = int(value)
    if selected not in allowed:
        raise ValueError(f"Invalid horizon {value}; allowed values: {allowed}")
    return [selected]


if __name__ == "__main__":
    main()
