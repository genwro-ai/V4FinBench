import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from v4finbench.data.io import read_horizon
from v4finbench.data.schema import HORIZONS, TARGET_COL, YEAR_COL
from v4finbench.data.splits import SplitConfig, temporal_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe the walk-forward splits and write cells.txt."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/temporal_splits"))
    parser.add_argument(
        "--horizons", type=int, nargs="+", choices=HORIZONS, default=HORIZONS
    )
    parser.add_argument("--availability-margin", type=int, default=0)
    return parser.parse_args()


def describe(idx: np.ndarray, years: np.ndarray, y: np.ndarray) -> dict:
    return {
        "years": [int(years[idx].min()), int(years[idx].max())] if len(idx) else [],
        "n_rows": int(len(idx)),
        "positives": int(y[idx].sum()),
    }


def main() -> None:
    args = parse_args()
    config = SplitConfig(availability_margin=args.availability_margin)
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"{'h':>2} {'Y':>5} {'train':>11} {'val':>11} {'tr_pos':>7} {'va_pos':>7} "
        f"{'te_pos':>7} {'te_rows':>9}"
    )
    summaries, cells = [], []
    for horizon in args.horizons:
        df = read_horizon(args.data_dir, horizon)
        years, y = df[YEAR_COL].to_numpy(), df[TARGET_COL].to_numpy()
        for split in temporal_splits(df, horizon, config, include_skipped=True):
            train = describe(split.train_idx, years, y)
            val = describe(split.val_idx, years, y)
            test = describe(split.test_idx, years, y)
            summaries.append(
                {
                    "horizon": horizon,
                    "test_year": split.test_year,
                    "skip_reason": split.skip_reason,
                    "train": train,
                    "val": val,
                    "test": test,
                }
            )
            flag = f"  SKIP ({split.skip_reason})" if split.skip_reason else ""
            print(
                f"{horizon:>2} {split.test_year:>5} "
                f"{'-'.join(map(str, train['years'])):>11} "
                f"{'-'.join(map(str, val['years'])):>11} "
                f"{train['positives']:>7} {val['positives']:>7} "
                f"{test['positives']:>7} {test['n_rows']:>9,}{flag}"
            )
            if split.skip_reason is None:
                cells.append(f"{horizon} {split.test_year}")

    (args.out / "split_definition.json").write_text(
        json.dumps({"config": asdict(config), "splits": summaries}, indent=2),
        encoding="utf-8",
    )
    (args.out / "cells.txt").write_text("\n".join(cells) + "\n", encoding="utf-8")
    print(f"\nWrote split_definition.json and cells.txt ({len(cells)} cells)")


if __name__ == "__main__":
    main()
