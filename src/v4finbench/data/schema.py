from dataclasses import dataclass
from pathlib import Path

KAGGLE_DATASET = "sebastiantomczak10/v4-group-corporate-bankruptcy"
KAGGLE_URL = (
    "https://www.kaggle.com/datasets/"
    "sebastiantomczak10/v4-group-corporate-bankruptcy/data"
)

UNLABELED_FILE = "company_years.parquet"
HORIZON_FILES = {
    0: "company_years_h1.parquet",
    1: "company_years_h2.parquet",
    2: "company_years_h3.parquet",
    3: "company_years_h4.parquet",
    4: "company_years_h5.parquet",
    5: "company_years_h6.parquet",
}


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path

    @property
    def unlabeled(self) -> Path:
        return self.data_dir / UNLABELED_FILE

    def horizon(self, horizon: int) -> Path:
        try:
            file_name = HORIZON_FILES[horizon]
        except KeyError as exc:
            raise ValueError(f"Unknown paper horizon: {horizon}") from exc
        return self.data_dir / file_name
