import pandas as pd
import pytest

from v4finbench.data.labels import create_horizon_labels

DISTRESSED = (-0.5, -0.5, 0.3)
HEALTHY = (0.5, 0.5, 2.0)


def _panel(firms: dict[str, dict[int, tuple[float, float, float]]]) -> pd.DataFrame:
    rows = [
        {
            "emis_id": firm,
            "year": year,
            "Equity/total_assets": equity,
            "EBITDA/total_assets": ebitda,
            "Current_assets/short_term_liabilities": current_ratio,
        }
        for firm, reports in firms.items()
        for year, (equity, ebitda, current_ratio) in reports.items()
    ]
    return pd.DataFrame(rows)


def _labels(df: pd.DataFrame, horizon: int, **kwargs) -> dict[tuple[str, int], int]:
    labeled = create_horizon_labels(df, horizon, **kwargs)
    return {
        (row.emis_id, row.year): row.main_label
        for row in labeled.itertuples(index=False)
    }


def _observed(df: pd.DataFrame, horizon: int, **kwargs) -> dict[tuple[str, int], int]:
    labeled = create_horizon_labels(df, horizon, **kwargs)
    return {
        (row.emis_id, row.year): row.label_observed_year
        for row in labeled.itertuples(index=False)
    }


def test_final_distressed_report_is_the_event() -> None:
    df = _panel({"d": {2016: HEALTHY, 2017: HEALTHY, 2018: DISTRESSED}})

    assert _labels(df, 1) == {("d", 2016): 0, ("d", 2017): 1}
    assert _labels(df, 2) == {("d", 2016): 1}


def test_gap_does_not_shift_positive_onto_another_year() -> None:
    df = _panel({"d": {2014: HEALTHY, 2015: HEALTHY, 2017: HEALTHY, 2018: DISTRESSED}})

    assert _labels(df, 1) == {("d", 2014): 0, ("d", 2015): 0, ("d", 2017): 1}
    assert _labels(df, 2) == {("d", 2014): 0, ("d", 2015): 0}
    assert _labels(df, 3) == {("d", 2014): 0, ("d", 2015): 1}


def test_within_mode_labels_every_row_in_the_window() -> None:
    df = _panel({"d": {2014: HEALTHY, 2015: HEALTHY, 2017: HEALTHY, 2018: DISTRESSED}})

    assert _labels(df, 2, mode="within") == {
        ("d", 2014): 0,
        ("d", 2015): 0,
        ("d", 2017): 1,
    }
    assert _labels(df, 3, mode="within") == {
        ("d", 2014): 0,
        ("d", 2015): 1,
        ("d", 2017): 1,
    }


def test_composite_criterion_requires_all_three_conditions() -> None:
    df = _panel({"p": {2019: HEALTHY, 2020: (0.1, -0.5, 0.3)}})

    assert _labels(df, 1) == {("p", 2019): 0}


def test_firm_still_reporting_in_terminal_year_has_no_event() -> None:
    df = _panel({"o": {2019: DISTRESSED, 2020: DISTRESSED, 2021: DISTRESSED}})

    assert _labels(df, 1) == {("o", 2019): 0}


def test_healthy_terminal_report_is_a_resolved_negative() -> None:
    df = _panel({"o": {2020: HEALTHY, 2021: HEALTHY}})

    assert _labels(df, 1) == {("o", 2020): 0}


def test_healthy_exit_is_censored_unless_kept() -> None:
    df = _panel({"c": {2015: HEALTHY, 2016: HEALTHY}})

    assert _labels(df, 2) == {}
    assert _labels(df, 2, drop_censored=False) == {("c", 2015): 0, ("c", 2016): 0}
    assert _observed(df, 2, drop_censored=False) == {
        ("c", 2015): pd.NA,
        ("c", 2016): pd.NA,
    }


def test_label_observed_year() -> None:
    df = _panel(
        {
            "d": {2014: HEALTHY, 2015: DISTRESSED, 2017: HEALTHY, 2018: DISTRESSED},
            "h": {2016: HEALTHY, 2017: HEALTHY},
        }
    )

    assert _observed(df, 1) == {
        ("d", 2014): 2017,  # 2015 report distressed: resolved by the 2017 report
        ("d", 2015): 2017,  # no report in 2016: resolved by the 2017 report
        ("d", 2017): 2018,  # positive: observed in the event year
        ("h", 2016): 2017,  # healthy report in the target year
    }


def test_duplicate_firm_years_are_rejected() -> None:
    df = _panel({"d": {2018: HEALTHY}})
    df = pd.concat([df, df])

    with pytest.raises(ValueError, match="duplicated"):
        create_horizon_labels(df, 1)


def test_horizon_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        create_horizon_labels(_panel({"d": {2018: HEALTHY}}), 0)
