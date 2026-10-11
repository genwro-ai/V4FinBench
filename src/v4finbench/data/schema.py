UNLABELED_FILE = "company_years.parquet"

FIRM_COL = "emis_id"
YEAR_COL = "year"
TARGET_COL = "main_label"
LABEL_YEAR_COL = "label_observed_year"

COUNTRIES = {0: "CZ", 1: "HU", 2: "PL", 3: "SK"}
HORIZONS = (1, 2, 3, 4, 5)
TERMINAL_YEAR = 2021


def horizon_file(horizon: int) -> str:
    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon {horizon}; expected one of {HORIZONS}")
    return f"company_years_h{horizon}.parquet"
