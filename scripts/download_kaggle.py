import argparse
import shutil
from pathlib import Path

from v4finbench.data.schema import KAGGLE_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download V4FinBench data from Kaggle."
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Install Kaggle support first: uv sync --extra kaggle"
        ) from exc

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    for path in downloaded.iterdir():
        if path.is_file():
            shutil.copy2(path, args.out / path.name)
    print(f"Downloaded {KAGGLE_DATASET} to {args.out}")


if __name__ == "__main__":
    main()
