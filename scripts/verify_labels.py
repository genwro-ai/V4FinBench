import argparse
from pathlib import Path

import pandas as pd

from v4finbench.data.labels import verify_horizon_labels
from v4finbench.data.schema import HORIZON_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify generated V4FinBench labels against reference files."
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing generated company_years_h*.parquet files.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing reference Kaggle company_years_h*.parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_horizon_files(args.generated, "generated")
    _require_horizon_files(args.reference, "reference")

    print(
        f"Comparing generated labels in {args.generated} "
        f"against reference labels in {args.reference} ...",
        flush=True,
    )
    results = verify_horizon_labels(args.generated, args.reference)
    for horizon, result in results.items():
        file_name = HORIZON_FILES[horizon]
        generated = pd.read_parquet(args.generated / file_name, columns=["main_label"])
        positives = int(generated["main_label"].sum())
        print(
            f"Verified paper horizon h={horizon}: {result} "
            f"({len(generated):,} rows, {positives:,} positives)",
            flush=True,
        )


def _require_horizon_files(path: Path, label: str) -> None:
    missing = [
        file_name
        for file_name in HORIZON_FILES.values()
        if not (path / file_name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} label file(s) in {path}: {', '.join(missing)}"
        )


if __name__ == "__main__":
    main()
