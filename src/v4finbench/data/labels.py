from typing import Literal

import numpy as np
import pandas as pd

from v4finbench.data.schema import (
    FIRM_COL,
    LABEL_YEAR_COL,
    TARGET_COL,
    TERMINAL_YEAR,
    YEAR_COL,
)

LabelMode = Literal["exact", "within"]


def meets_distress_criterion(df: pd.DataFrame) -> pd.Series:
    return (
        (df["Equity/total_assets"] < 0)
        & (df["EBITDA/total_assets"] < 0)
        & (df["Current_assets/short_term_liabilities"] <= 0.6)
    )


def create_horizon_labels(
    df: pd.DataFrame,
    horizon: int,
    mode: LabelMode = "exact",
    drop_censored: bool = True,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if mode not in ("exact", "within"):
        raise ValueError(f"Unknown label mode: {mode}")
    duplicated = df.duplicated([FIRM_COL, YEAR_COL])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicated firm-year rows")

    firm = df[FIRM_COL]
    year = df[YEAR_COL].to_numpy()
    distressed = meets_distress_criterion(df).to_numpy()

    last_year = df.groupby(firm)[YEAR_COL].transform("max").to_numpy()
    final_distressed = (
        pd.Series(distressed & (year == last_year), index=df.index)
        .groupby(firm)
        .transform("any")
        .to_numpy()
    )
    has_event = final_distressed & (last_year < TERMINAL_YEAR)
    target_year = year + horizon

    if mode == "exact":
        positive = has_event & (target_year == last_year)
    else:
        positive = has_event & (year < last_year) & (last_year <= target_year)

    negative = (target_year < last_year) | (
        ~has_event
        & (target_year == last_year)
        & ~((last_year == TERMINAL_YEAR) & final_distressed)
    )
    if not drop_censored:
        negative |= (
            ~has_event
            & (target_year > last_year)
            & (target_year <= TERMINAL_YEAR)
        )

    keep = positive | negative
    labeled = df.loc[keep].copy()
    labeled[TARGET_COL] = positive[keep].astype(np.int8)
    labeled[LABEL_YEAR_COL] = _label_observed_year(
        labeled[FIRM_COL].to_numpy(),
        target_year[keep],
        np.where(positive[keep], last_year[keep], np.nan),
        pd.DataFrame(
            {FIRM_COL: firm.to_numpy(), "_report_year": year, "_distressed": distressed}
        ),
    )
    return labeled.reset_index(drop=True)


def _label_observed_year(
    firms: np.ndarray,
    target_year: np.ndarray,
    event_year: np.ndarray,
    reports: pd.DataFrame,
) -> pd.arrays.IntegerArray:
    queries = pd.DataFrame(
        {FIRM_COL: firms, "_target_year": target_year, "_row": np.arange(len(firms))}
    )
    at_target = queries.merge(
        reports,
        left_on=[FIRM_COL, "_target_year"],
        right_on=[FIRM_COL, "_report_year"],
        how="left",
        validate="many_to_one",
    )
    healthy_at_target = (at_target["_distressed"] == False).to_numpy()  # noqa: E712

    next_report = pd.merge_asof(
        queries.sort_values("_target_year"),
        reports.sort_values("_report_year"),
        left_on="_target_year",
        right_on="_report_year",
        by=FIRM_COL,
        direction="forward",
        allow_exact_matches=False,
    ).sort_values("_row")
    next_year = next_report["_report_year"].to_numpy(dtype=float)

    observed = np.where(healthy_at_target, target_year, next_year)
    observed = np.where(np.isnan(event_year), observed, event_year)
    return pd.array(observed, dtype="Int64")
