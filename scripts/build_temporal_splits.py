"""Generate chronological (out-of-time) evaluation splits for V4FinBench.

Walk-forward design with a label-availability rule. For each test
(statement) year Y:

    test  : all statements from year Y, labeled at every horizon h with
            observable outcomes (cells without positives are skipped)
    pool  : statements from years before Y whose horizon-h label was
            KNOWABLE by the end of Y (see below)
    val   : the two most recent statement years present in the pool
            (threshold + hyperparameter selection)
    train : all earlier pool rows

Label availability. V4FinBench horizons are report-indexed: the horizon-h
label of a statement refers to the firm's h-th subsequent report, which can
be more than h calendar years later when firms skip years. For every row we
therefore derive label_available_year, the year in which the label's
deciding report was filed:

    positive row: the firm's final (distressed) report year
    negative row: the year of the h-th subsequent report. If that report
                  does not exist (trailing rows of firms that stopped
                  reporting), the label rests on unobserved continuation
                  and the row is never admitted to training or validation.

Only rows with label_available_year <= Y enter training or validation, so
the model and its decision threshold are built exclusively from information
that existed at prediction time. Test rows are exempt by design: their
labels are the future being predicted. A row's admission depends on the
test year, not on the row: a row excluded for Y=2012 may be admissible for
Y=2014.

Timing assumption. The cutoff "available <= Y" treats prediction as
happening once the year-Y statements exist, i.e., after fiscal year Y has
closed (annual reports for fiscal year Y are filed afterwards, so scoring
them implies all reports filed through year Y are observable). Under this
reading, labels resolved by reports filed during year Y are legitimately
known at prediction time. Pass --availability-margin 1 for the stricter
convention "available < Y", which shifts every training and validation
window one further year into the past.

Entity grouping. Availability is computed over the same company-name
groups used by the label construction itself, not over emis_id: each
released label's deciding report is defined by its name-group sequence, so
availability must follow the same grouping to describe the right report.
The 187 name-collision groups (0.15% of rows) are handled conservatively
by the max-aggregation and the +h floor below.

Output layout mirrors data/folds so run_baselines.py and
run_temporal_baseline.py can consume it via --folds-dir/--splits-dir with
the origin index in place of the fold index.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v4finbench.data.schema import HORIZON_FILES, UNLABELED_FILE

FIRST_TEST_YEAR = 2012
LAST_TEST_YEAR = 2020
VAL_N_YEARS = 2

MIN_TEST_POSITIVES = 30
MIN_TRAIN_POSITIVES = 30
MIN_VAL_POSITIVES = 15


@dataclass(frozen=True)
class OriginResult:
    test_year: int
    train_years: str
    val_years: list[int]
    emitted: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-panel", type=Path,
                        default=Path("data/raw") / UNLABELED_FILE)
    parser.add_argument("--out", type=Path, default=Path("data/temporal_splits"))
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--company-col", default="company")
    parser.add_argument("--target-col", default="main_label")
    parser.add_argument(
        "--availability-margin", type=int, default=0,
        help="Admit rows with label_available_year <= test_year - margin. "
        "0 (default) assumes prediction after fiscal year Y closes; "
        "1 is the stricter available-before-Y convention.",
    )
    return parser.parse_args()


def label_available_year(
    task: pd.DataFrame,
    base: pd.DataFrame,
    horizon: int,
    company_col: str,
    year_col: str,
    target_col: str,
) -> np.ndarray:
    """Year in which each task row's horizon-h label became observable."""
    base = base.sort_values([company_col, year_col])
    grouped = base.groupby(company_col)[year_col]
    lookup = pd.DataFrame(
        {
            company_col: base[company_col].to_numpy(),
            year_col: base[year_col].to_numpy(),
            # year of the h-th subsequent report (NaN if it does not exist)
            "_next_h": grouped.shift(-horizon).to_numpy(),
            "_final": grouped.transform("max").to_numpy(),
        }
    )
    # Negatives whose h-th subsequent report does not exist are censored:
    # their label rests on the firm never reporting again, which is not
    # observable at any prediction origin. NEVER_AVAILABLE keeps them out
    # of every training and validation pool.
    NEVER_AVAILABLE = 9999
    lookup["_next_h"] = lookup["_next_h"].fillna(NEVER_AVAILABLE)
    # A small number of (company, year) pairs are duplicated in the panel
    # (0.15% of rows, name collisions between distinct firms). Aggregate
    # conservatively with max(), i.e. the latest possible availability, so
    # an ambiguous row can only be excluded too long, never admitted early.
    lookup = (
        lookup.groupby([company_col, year_col], as_index=False)
        .agg(_next_h=("_next_h", "max"), _final=("_final", "max"))
    )

    merged = task[[company_col, year_col, target_col]].merge(
        lookup, on=[company_col, year_col], how="left", validate="many_to_one"
    )
    if merged["_final"].isna().any():
        missing = int(merged["_final"].isna().sum())
        raise ValueError(f"{missing} task rows not found in the base panel.")

    available = np.where(
        merged[target_col].to_numpy() == 1,
        merged["_final"].to_numpy(),
        merged["_next_h"].to_numpy(),
    )
    # Floor at statement year + h: h subsequent annual reports cannot
    # resolve in fewer than h calendar years. This also neutralizes the
    # name-collision groups above, whose merged sequences contain several
    # reports per calendar year and would otherwise resolve implausibly
    # early.
    available = np.maximum(
        available, merged[year_col].to_numpy() + horizon
    )
    return available.astype(int)


def _class_counts(y: np.ndarray) -> dict[str, int]:
    return {"0": int((y == 0).sum()), "1": int((y == 1).sum())}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(args.base_panel,
                           columns=[args.company_col, args.year_col])

    definition: dict[str, object] = {
        "description": "Walk-forward out-of-time splits with a "
                       "label-availability rule: train/validation admit only "
                       "rows whose horizon label was observable by the end "
                       "of the test year.",
        "design": "walk_forward_label_availability",
        "availability_margin": args.availability_margin,
        "test_years": list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)),
        "val_n_years": VAL_N_YEARS,
        "min_positives": {"train": MIN_TRAIN_POSITIVES,
                          "val": MIN_VAL_POSITIVES,
                          "test": MIN_TEST_POSITIVES},
        "horizons": {},
    }

    print(f"{'h':>2} {'Y':>5} {'train':>14} {'val':>11} "
          f"{'tr_pos':>7} {'va_pos':>7} {'te_pos':>7} {'te_rows':>9}")
    print("-" * 78)

    all_cells: list[str] = []

    for horizon, file_name in HORIZON_FILES.items():
        task = pd.read_parquet(
            args.data_dir / file_name,
            columns=[args.company_col, args.year_col, args.target_col],
        )
        years = task[args.year_col].to_numpy()
        y = task[args.target_col].astype(int).to_numpy()
        available = label_available_year(
            task, base, horizon,
            args.company_col, args.year_col, args.target_col,
        )

        horizon_dir = args.out / f"h{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        meta_splits: dict[str, object] = {}
        emitted: list[int] = []

        for oi, test_year in enumerate(
            range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)
        ):
            test_mask = years == test_year
            cutoff = test_year - args.availability_margin
            pool_mask = (years < test_year) & (available <= cutoff)

            pool_years = np.unique(years[pool_mask])
            if len(pool_years) < VAL_N_YEARS + 1:
                continue
            val_years = pool_years[-VAL_N_YEARS:]
            val_mask = pool_mask & np.isin(years, val_years)
            train_mask = pool_mask & (years < val_years.min())

            tr_pos = int(y[train_mask].sum())
            va_pos = int(y[val_mask].sum())
            te_pos = int(y[test_mask].sum())

            reason = ""
            if te_pos < MIN_TEST_POSITIVES:
                reason = "few test positives"
            elif tr_pos < MIN_TRAIN_POSITIVES:
                reason = "few train positives"
            elif va_pos < MIN_VAL_POSITIVES:
                reason = "few val positives"

            train_range = f"<= {int(years[train_mask].max())}" if train_mask.any() else "-"
            val_range = f"{val_years.min()}-{val_years.max()}"
            flag = f"  SKIP ({reason})" if reason else ""
            print(f"{horizon:>2} {test_year:>5} {train_range:>14} {val_range:>11} "
                  f"{tr_pos:>7} {va_pos:>7} {te_pos:>7} "
                  f"{int(test_mask.sum()):>9,}{flag}")

            if reason:
                continue

            np.savetxt(horizon_dir / f"split_{oi}_train_idx.txt",
                       np.where(train_mask)[0], fmt="%d")
            np.savetxt(horizon_dir / f"split_{oi}_val_idx.txt",
                       np.where(val_mask)[0], fmt="%d")
            np.savetxt(horizon_dir / f"split_{oi}_test_idx.txt",
                       np.where(test_mask)[0], fmt="%d")
            emitted.append(test_year)
            all_cells.append(f"{horizon} {oi} {test_year}")
            meta_splits[str(oi)] = {
                "origin": str(test_year),
                "train": {"years": train_range,
                          "n_rows": int(train_mask.sum()),
                          "class_counts": _class_counts(y[train_mask])},
                "val": {"years": [int(v) for v in val_years],
                        "n_rows": int(val_mask.sum()),
                        "class_counts": _class_counts(y[val_mask])},
                "test": {"years": [test_year],
                         "n_rows": int(test_mask.sum()),
                         "class_counts": _class_counts(y[test_mask])},
            }

        (horizon_dir / "metadata.json").write_text(
            json.dumps({"horizon": horizon, "splits": meta_splits}, indent=2),
            encoding="utf-8",
        )
        definition["horizons"][str(horizon)] = {"emitted_test_years": emitted}

    (args.out / "split_definition.json").write_text(
        json.dumps(definition, indent=2), encoding="utf-8"
    )
    # Manifest of emitted cells ("horizon origin_index test_year" per line)
    # for array schedulers: the cell grid is triangular and depends on the
    # thresholds above, so consumers must read it rather than assume a
    # rectangle.
    (args.out / "cells.txt").write_text(
        "\n".join(all_cells) + "\n", encoding="utf-8"
    )
    print(f"\nWrote temporal splits to {args.out}")
    print(f"Split definition: {args.out / 'split_definition.json'}")
    print(f"Cell manifest: {args.out / 'cells.txt'} ({len(all_cells)} cells)")


if __name__ == "__main__":
    main()
