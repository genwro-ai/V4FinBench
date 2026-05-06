import math

import pandas as pd

from v4finbench.llama.formatting import (
    SYSTEM_PROMPT,
    dataframe_to_llama_records,
    format_company_data,
    horizon_description,
    render_system_prompt,
)
from v4finbench.llama.inference import extract_yes_no, normalize_yes_no_logprobs
from v4finbench.llama.metrics import (
    hard_prediction_metrics,
    threshold_probability_predictions,
)
from v4finbench.llama.sampling import LlamaSplitConfig, create_llama_train_test_splits


def test_format_company_data_uses_available_columns_and_mappings():
    row = {
        "country": 0,
        "state": 7,
        "legal_form": 2,
        "year": 2020,
        "Cash/total_assets": 0.12345,
        "main_label": 1,
    }

    formatted = format_company_data(row)

    assert "country=0 (Poland)" in formatted
    assert "state=7 (Mazowieckie)" in formatted
    assert "legal_form=2 (Limited Liability Company)" in formatted
    assert "Cash/total_assets=0.123" in formatted


def test_dataframe_to_llama_records_adds_fixed_prompt_and_yes_no_label():
    df = pd.DataFrame(
        [
            {"year": 2020, "country": 0, "emis_id": "a", "main_label": 1},
            {"year": 2021, "country": 1, "emis_id": "b", "main_label": 0},
        ]
    )

    records = dataframe_to_llama_records(df)

    assert records["system"].tolist() == [SYSTEM_PROMPT, SYSTEM_PROMPT]
    assert records["assistant"].tolist() == ["YES", "NO"]
    assert records["main_label"].tolist() == [1, 0]


def test_llama_prompt_is_parameterized_by_horizon():
    prompt = render_system_prompt(
        "Predict whether the company will go bankrupt {horizon_description}.",
        horizon=3,
    )
    df = pd.DataFrame([{"year": 2020, "country": 0, "emis_id": "a", "main_label": 1}])

    records = dataframe_to_llama_records(df, system_prompt=prompt)

    assert "go bankrupt" in records.loc[0, "system"]
    assert "3 years after the observed reporting year" in records.loc[0, "system"]


def test_horizon_description_matches_paper_convention():
    assert horizon_description(0) == "in the current reporting year"
    assert horizon_description(1) == "one year after the observed reporting year"
    assert horizon_description(5) == "5 years after the observed reporting year"


def test_llama_sampling_is_disjoint_and_has_requested_positive_ratio():
    df = pd.DataFrame(
        {
            "company": [f"c{i}" for i in range(200)],
            "year": [2020] * 200,
            "main_label": [1] * 40 + [0] * 160,
        }
    )
    config = LlamaSplitConfig(
        train_size=20,
        test_size=40,
        min_positive_ratio_train=0.25,
        min_positive_ratio_test=0.25,
        seed=42,
    )

    train_df, test_df = create_llama_train_test_splits(df, config)

    assert len(train_df) == 20
    assert len(test_df) == 40
    assert train_df["main_label"].sum() == 5
    assert test_df["main_label"].sum() == 10
    train_keys = set(zip(train_df["company"], train_df["year"], strict=True))
    test_keys = set(zip(test_df["company"], test_df["year"], strict=True))
    assert train_keys.isdisjoint(test_keys)


def test_llama_sampling_uses_available_rows_when_positive_ratio_is_impossible():
    df = pd.DataFrame(
        {
            "company": [f"c{i}" for i in range(60)],
            "year": [2020] * 60,
            "main_label": [1] * 8 + [0] * 52,
        }
    )
    config = LlamaSplitConfig(
        train_size=20,
        test_size=40,
        min_positive_ratio_train=0.50,
        min_positive_ratio_test=0.50,
        seed=42,
    )

    train_df, test_df = create_llama_train_test_splits(df, config)

    assert len(train_df) == 20
    assert len(test_df) == 40
    assert train_df["main_label"].sum() == 8
    assert test_df["main_label"].sum() == 0
    combined = pd.concat([train_df, test_df])
    assert len(combined.drop_duplicates(["company", "year"])) == 60


def test_extract_yes_no_takes_first_valid_answer():
    assert extract_yes_no("YES.") == "YES"
    assert extract_yes_no("No, it will not.") == "NO"
    assert extract_yes_no("NO then YES") == "NO"
    assert extract_yes_no("unclear") is None


def test_normalize_yes_no_logprobs_is_stable():
    p_yes, p_no = normalize_yes_no_logprobs(-1000.0, -1001.0)

    assert math.isclose(p_yes + p_no, 1.0)
    assert p_yes > p_no


def test_llama_metrics_score_hard_and_probability_predictions():
    df = pd.DataFrame(
        {
            "assistant": ["YES", "NO", "YES", "NO"],
            "prediction": ["YES", "NO", "UNKNOWN", "YES"],
            "p_yes": [0.9, 0.1, 0.8, 0.4],
        }
    )

    hard = hard_prediction_metrics(df)
    thresholded, probability = threshold_probability_predictions(df)

    assert hard["known_predictions"] == 3
    assert hard["unknown_predictions"] == 1
    assert probability["f1"] == 1.0
    assert set(thresholded["prediction"]) == {"YES", "NO"}


def test_llama_hard_metrics_allow_all_unknown_predictions():
    df = pd.DataFrame(
        {
            "assistant": ["YES", "NO"],
            "prediction": ["UNKNOWN", "UNKNOWN"],
        }
    )

    metrics = hard_prediction_metrics(df)

    assert metrics["known_predictions"] == 0
    assert metrics["unknown_predictions"] == 2
    assert math.isnan(metrics["f1"])
