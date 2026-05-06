from __future__ import annotations

import pandas as pd

from v4finbench.data.labels import create_horizon_labels


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company": [
                "distressed",
                "distressed",
                "distressed",
                "healthy",
                "healthy",
                "open_2021",
                "open_2021",
                "short",
            ],
            "year": [2018, 2019, 2020, 2019, 2020, 2020, 2021, 2020],
            "Equity/total_assets": [-0.1, -0.2, -0.3, 0.2, 0.3, -1.0, -1.0, -0.2],
            "EBITDA/total_assets": [-0.1, -0.2, -0.3, -0.2, -0.3, -1.0, -1.0, -0.2],
            "Current_assets/short_term_liabilities": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.3,
                0.3,
                0.5,
            ],
        }
    )


def test_current_horizon_labels_final_distress_report() -> None:
    labeled = create_horizon_labels(_base_df(), horizon=0)
    positive = labeled[labeled["main_label"] == 1]

    assert positive[["company", "year"]].to_dict("records") == [
        {"company": "distressed", "year": 2020},
        {"company": "short", "year": 2020},
    ]


def test_future_horizon_moves_positive_label_back() -> None:
    labeled = create_horizon_labels(_base_df(), horizon=1)
    positive = labeled[labeled["main_label"] == 1]

    assert positive[["company", "year"]].to_dict("records") == [
        {"company": "distressed", "year": 2019},
    ]


def test_composite_condition_requires_all_three_criteria() -> None:
    df = pd.DataFrame(
        {
            "company": ["partial", "partial"],
            "year": [2019, 2020],
            "Equity/total_assets": [-0.1, 0.1],
            "EBITDA/total_assets": [-0.1, -0.1],
            "Current_assets/short_term_liabilities": [0.5, 0.5],
        }
    )

    labeled = create_horizon_labels(df, horizon=0)

    assert labeled["main_label"].sum() == 0


def test_final_2021_report_is_not_labeled_positive() -> None:
    labeled = create_horizon_labels(_base_df(), horizon=0)
    open_company = labeled[labeled["company"] == "open_2021"]

    assert open_company["year"].tolist() == [2020]
    assert open_company["main_label"].tolist() == [0]

