from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from tqdm.auto import tqdm

from v4finbench.data.schema import HORIZON_FILES


@dataclass(frozen=True)
class LabelConfig:
    group_col: str = "company"
    year_col: str = "year"
    target_col: str = "main_label"
    equity_col: str = "Equity/total_assets"
    ebitda_col: str = "EBITDA/total_assets"
    current_ratio_col: str = "Current_assets/short_term_liabilities"
    terminal_year: int = 2021
    current_ratio_threshold: float = 0.6


DEFAULT_LABEL_CONFIG = LabelConfig()


def is_distressed_final_report(row: pd.Series, config: LabelConfig) -> bool:
    return bool(
        (row[config.equity_col] < 0)
        and (row[config.ebitda_col] < 0)
        and (row[config.current_ratio_col] <= config.current_ratio_threshold)
    )


def create_horizon_labels(
    df: pd.DataFrame,
    horizon: int,
    config: LabelConfig | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Create one horizon-specific binary distress task.

    ``horizon`` uses the paper convention: ``0`` means current-year distress and
    ``5`` means five-year-ahead distress. The output file name mapping to Kaggle
    h1..h6 files is handled separately in ``schema.HORIZON_FILES``.
    """
    config = config or DEFAULT_LABEL_CONFIG
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    _validate_required_columns(df, config)

    labeled_groups: list[pd.DataFrame] = []
    grouped = df.groupby(config.group_col)
    groups = tqdm(
        grouped,
        total=grouped.ngroups,
        desc=f"Building labels h={horizon}",
        unit="company",
        disable=not show_progress,
    )
    for _, group in groups:
        labeled = _process_company_group(group, horizon, config)
        if labeled is not None and not labeled.empty:
            labeled_groups.append(labeled)

    if not labeled_groups:
        return df.iloc[0:0].copy().assign(**{config.target_col: pd.Series(dtype=int)})
    return pd.concat(labeled_groups, ignore_index=True)


def create_all_horizon_labels(
    df: pd.DataFrame,
    n_horizons: int = 6,
    config: LabelConfig | None = None,
    show_progress: bool = False,
) -> dict[int, pd.DataFrame]:
    return {
        horizon: create_horizon_labels(df, horizon, config, show_progress)
        for horizon in range(n_horizons)
    }


def write_horizon_labels(
    df: pd.DataFrame,
    output_dir: str | Path,
    n_horizons: int = 6,
    config: LabelConfig | None = None,
    show_progress: bool = False,
) -> dict[int, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written: dict[int, Path] = {}
    labeled_by_horizon = create_all_horizon_labels(
        df,
        n_horizons,
        config,
        show_progress,
    )
    for horizon, labeled in labeled_by_horizon.items():
        file_name = HORIZON_FILES[horizon]
        path = output_path / file_name
        labeled.to_parquet(path, index=False)
        written[horizon] = path
    return written


def verify_horizon_labels(
    generated_dir: str | Path,
    reference_dir: str | Path,
    horizons: list[int] | None = None,
) -> dict[int, str]:
    horizons = horizons or sorted(HORIZON_FILES)
    results = {}
    for horizon in horizons:
        file_name = HORIZON_FILES[horizon]
        generated = pd.read_parquet(Path(generated_dir) / file_name)
        reference = pd.read_parquet(Path(reference_dir) / file_name)
        assert_frame_equal(generated, reference, check_dtype=False)
        results[horizon] = "ok"
    return results


def _process_company_group(
    group: pd.DataFrame,
    horizon: int,
    config: LabelConfig,
) -> pd.DataFrame | None:
    group = group.sort_values(config.year_col)
    last_report = group.iloc[-1]

    if last_report[config.year_col] == config.terminal_year:
        return _negative_until_cutoff(group, horizon, config, drop_terminal_year=True)

    if not is_distressed_final_report(last_report, config):
        return _negative_until_cutoff(group, horizon, config, drop_terminal_year=False)

    if len(group) <= horizon:
        return None

    labeled = group.iloc[: len(group) - horizon] if horizon > 0 else group
    if labeled.empty:
        return None

    labeled = labeled.copy()
    labeled[config.target_col] = 0
    labeled.loc[labeled.index[-1], config.target_col] = 1
    return labeled


def _negative_until_cutoff(
    group: pd.DataFrame,
    horizon: int,
    config: LabelConfig,
    drop_terminal_year: bool,
) -> pd.DataFrame | None:
    cutoff_year = config.terminal_year - horizon
    if drop_terminal_year:
        group = group[group[config.year_col] != config.terminal_year]
    labeled = group[group[config.year_col] <= cutoff_year]
    if labeled.empty:
        return None
    labeled = labeled.copy()
    labeled[config.target_col] = 0
    return labeled


def _validate_required_columns(df: pd.DataFrame, config: LabelConfig) -> None:
    required = {
        config.group_col,
        config.year_col,
        config.equity_col,
        config.ebitda_col,
        config.current_ratio_col,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required label column(s): {sorted(missing)}")
