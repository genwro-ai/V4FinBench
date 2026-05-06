from collections.abc import Mapping
from typing import Any

import pandas as pd

DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a bankruptcy prediction model.\n"
    "Based on structured company and financial data, predict whether the company "
    "will go bankrupt {horizon_description}.\n"
    "Respond with only one word: YES or NO."
)
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
    horizon=0,
    horizon_years=0,
    horizon_description="in the current reporting year",
)

VALUE_MAPPINGS: dict[str, dict[Any, str]] = {
    "country": {
        0: "Poland",
        1: "Czech Republic",
        2: "Slovakia",
        3: "Hungary",
    },
    "has_multiple_industries": {1: "Multiple Industries", 0: "Single Industry"},
    "sector_1": {
        1: "Construction",
        2: "Manufacturing",
        3: "Retail Trade",
        4: "Wholesale Trade",
        5: "Transportation",
        6: "Power",
    },
    "sector_2": {
        1: "Construction",
        2: "Manufacturing",
        3: "Retail Trade",
        4: "Wholesale Trade",
        5: "Transportation",
        6: "Power",
    },
    "sector_3": {
        1: "Construction",
        2: "Manufacturing",
        3: "Retail Trade",
        4: "Wholesale Trade",
        5: "Transportation",
        6: "Power",
    },
    "number_of_employees": {
        0: "No data",
        1: "1-9 employees",
        2: "10-49 employees",
        3: "50-249 employees",
        4: ">250 employees",
    },
    "legal_form": {
        1: "Public Limited Company",
        2: "Limited Liability Company",
        3: "Other non-liability limited",
        4: "Limited Liability Partnership",
    },
    "incorporation_date_1": {
        1: "0-2y",
        2: "3-4y",
        3: "5-24y",
        4: ">24y",
    },
    "incorporation_date_2": {
        1: "0-1y",
        2: "1-2y",
        3: "3-5y",
        4: "6-9y",
        5: "10-19y",
        6: ">19y",
    },
    "operational_status": {
        0: "Liquidation, Under Legal Investigation, Closed",
        1: "Active",
    },
}

STATE_MAPPINGS: dict[str, dict[int, str]] = {
    "Czech Republic": {
        1: "Jihocesky",
        2: "Jihomoravsky",
        3: "Karlovarsky",
        4: "Kraj Vysocina",
        5: "Kralovehradecky",
        6: "Liberecky",
        7: "Moravskoslezsky",
        8: "Olomoucky",
        9: "Pardubicky",
        10: "Plzensky",
        11: "Stredocesky",
        12: "Prague",
        13: "Ustecky",
        14: "Zlinsky",
    },
    "Poland": {
        1: "Dolnoslaskie",
        2: "Kujawsko-Pomorskie",
        3: "Lodzkie",
        4: "Lubelskie",
        5: "Lubuskie",
        6: "Malopolskie",
        7: "Mazowieckie",
        8: "Opolskie",
        9: "Podkarpackie",
        10: "Podlaskie",
        11: "Pomorskie",
        12: "Slaskie",
        13: "Swietokrzyskie",
        14: "Warminsko-Mazurskie",
        15: "Wielkopolskie",
        16: "Zachodniopomorskie",
        17: "Others",
    },
    "Hungary": {
        1: "Bacs-Kiskun",
        2: "Baranya",
        3: "Bekes",
        4: "Borsod-Abauj-Zemplen",
        5: "Budapest",
        6: "Csongrad",
        7: "Fejer",
        8: "Gyor-Moson-Sopron",
        9: "Hajdu-Bihar",
        10: "Heves",
        11: "Jasz-Nagykun-Szolnok",
        12: "Komarom-Esztergom",
        13: "Nograd",
        14: "Pest",
        15: "Somogy",
        16: "Szabolcs-Szatmar-Bereg",
        17: "Tolna",
        18: "Vas",
        19: "Veszprem",
        20: "Zala",
    },
    "Slovakia": {
        1: "Banskobystricky",
        2: "Bratislavsky",
        3: "Kosicky",
        4: "Nitriansky",
        5: "Presovsky",
        6: "Trenciansky",
        7: "Trnavsky",
        8: "Zilina",
    },
}

METRIC_GROUPS: dict[str, list[str]] = {
    "Company Info": [
        "country",
        "state",
        "number_of_employees",
        "legal_form",
        "primary_naics_encoded",
        "naics_2digit",
        "naics_3digit",
        "has_multiple_industries",
        "secondary_naics_encoded",
        "sector_1",
        "sector_2",
        "sector_3",
        "incorporation_date_1",
        "incorporation_date_2",
        "operational_status",
        "year",
    ],
    "Text Analysis": ["text_analysis"],
    "Liquidity": [
        "Current_assets-inventories-receivables/short_term_liabilities",
        "Working_capital",
        "Cash/total_assets",
        "Inventories/working_capital",
        "Working_capital/equity",
        "Current_assets/sales",
        "Quick_assets/sales",
        "Cash/sales",
        "Cash/total_operating_revenue",
        "Current_assets/total_operating_revenue",
    ],
    "Profitability": [
        "Net_profit/total_operating_revenue",
        "Gross_profit/total_operating_revenue",
        "Net_profit/equity",
        "EBIT/equity",
        "EBIT/financial_costs",
        "Net_profit/fixed_assets",
        "Net_profit/inventories",
        "Retained_profit/total_assets",
        "Retained_profit/short_term_liabilities",
        "Net_profits/fixed_assets",
        "Net_profits/equity",
    ],
    "Leverage": [
        "Total_liabilities-cash/total_operating_revenue",
        "Total_liabilities-cash/EBITDA",
        "Total_liabilities/total_assets",
        "Equity/total_liabilities",
        "Equity/total_assets",
        "Equity/long_term_liabilities",
        "Long_term_liabilities/equity",
        "Short_term_liabilities/total_assets",
        "Current_liabilities/total_liabilities",
        "Current_liabilities/equity",
        "Current_liabilities/assets",
        "Long_term_liabilities/current_assets",
        "Equity-share_capital/fixed_assets",
        "Constant_capital/total_assets",
        "Fixed_assets/long_term_liabilities",
    ],
    "Efficiency": [
        "Receivables_turnover_days",
        "Short_term_liabilities_turnover_days",
        "Operating_cycle",
        "Cash_conversion_cycle",
        "Total_operating_revenue/total_assets",
        "Total_operating_revenue/fixed_assets",
        "Total_operating_revenue/inventories",
        "Total_operating_revenue/receivables",
        "Total_operating_revenue/short_term_liabilities",
        "Inventories/total_operating_revenue",
        "Inventory/current_liabilities",
        "Inventories+receivables/equity",
        "Inventory+receivables/equity",
    ],
    "Growth": [
        "Operating_revenue_growth",
        "Total_assets_growth",
        "Operating_profit_growth",
        "Net_profit_growth",
        "Current_assets_growth",
        "Inventories_growth",
        "Receivables_growth",
        "Short_term_liabilities_growth",
    ],
    "Revenue & Sales": [
        "Revenue/current_assets",
        "Revenue/fixed_assets",
        "Revenue/current_liabilities",
        "Revenue/long_term_liabilities",
        "Revenue/total_liabilities",
        "Revenue/assets_Sector",
        "Revenue/fixed_assets_Sector",
    ],
    "Sector Comparisons": [
        "Net_profits/sales_Sector",
        "Net_profits/assets_Sector",
        "Net_profits/fixed_assets_Sector",
        "Net_profits/abs_equity_Sector",
        "Net_profits/current_assets_Sector",
        "ST_receivables_investments/current_liabilities_Sector",
        "ST_financial_assets/current_liabilities_Sector",
        "Inventories*365/revenue_Sector",
        "Receivables*365/revenue_Sector",
        "Current_liabilities*365/revenue_Sector",
        "Cash_conversion_cycle_Sector",
        "Operating_cycle_Sector",
    ],
    "Logarithmic Metrics": [
        "Logarithm_of_total_assets",
        "Logarithm_of_total_operating_revenue",
        "Logarithm_of_total_assets/GDP",
        "Log_current_assets",
        "Log_total_liabilities",
        "Log_revenue/gdp",
        "Log_operating_profit/gdp",
        "Log_net_profit/gdp",
    ],
    "Cash Flow & Expenses": [
        "Cash_flow/total_debt",
        "EBITDA/cash_flow",
        "Cash_flow/sales",
        "Interest_expense/revenue",
        "Operating_expenses/sales",
        "Operational_expenses/short_term_liabilities",
        "Operational_expenses/total_liabilities",
    ],
    "Equity & Sales": [
        "Equity/sales",
        "Equity_ratio_classification",
    ],
    "Risk Flags": ["Insolvency_flag", "Loss_flag"],
}

CATEGORICAL_COLUMNS = {
    "country",
    "state",
    "primary_naics_encoded",
    "naics_2digit",
    "naics_3digit",
    "has_multiple_industries",
    "secondary_naics_encoded",
    "sector_1",
    "sector_2",
    "sector_3",
    "legal_form",
    "number_of_employees",
    "incorporation_date_1",
    "incorporation_date_2",
    "operational_status",
}


def format_llama_value(column: str, value: Any, country: str | None = None) -> Any:
    if column in VALUE_MAPPINGS and value in VALUE_MAPPINGS[column]:
        return f"{value} ({VALUE_MAPPINGS[column][value]})"
    if column == "state" and country in STATE_MAPPINGS:
        if value in STATE_MAPPINGS[country]:
            return f"{value} ({STATE_MAPPINGS[country][value]})"
        return value
    return value


def format_company_data(row: Mapping[str, Any]) -> str:
    country_code = row.get("country")
    country_name = VALUE_MAPPINGS["country"].get(country_code)
    groups = []
    for group_name, columns in METRIC_GROUPS.items():
        values = []
        for column in columns:
            if column not in row or pd.isna(row[column]):
                continue
            value = row[column]
            formatted_value = format_llama_value(column, value, country_name)
            if isinstance(value, int | float) and column not in CATEGORICAL_COLUMNS:
                if abs(value) >= 1000:
                    values.append(f"{column}={value:,.2f}")
                else:
                    values.append(f"{column}={value:.3f}")
            else:
                values.append(f"{column}={formatted_value}")
        if values:
            groups.append(f"{group_name}: {', '.join(values)}")
    return " | ".join(groups)


def row_to_llama_record(
    row: Mapping[str, Any],
    target_col: str = "main_label",
    include_label: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    record = {
        "year": row.get("year", ""),
        "country": row.get("country", ""),
        "emis_id": row.get("emis_id", ""),
        "system": system_prompt,
        "user": format_company_data(row),
    }
    if include_label:
        label = int(row[target_col])
        record[target_col] = label
        record["assistant"] = "YES" if label == 1 else "NO"
    return record


def dataframe_to_llama_records(
    df: pd.DataFrame,
    target_col: str = "main_label",
    include_label: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
) -> pd.DataFrame:
    records = [
        row_to_llama_record(
            row.to_dict(),
            target_col=target_col,
            include_label=include_label,
            system_prompt=system_prompt,
        )
        for _, row in df.iterrows()
    ]
    return pd.DataFrame(records)


def render_system_prompt(template: str, *, horizon: int) -> str:
    return template.format(
        horizon=horizon,
        horizon_years=horizon,
        horizon_description=horizon_description(horizon),
    )


def horizon_description(horizon: int) -> str:
    if horizon == 0:
        return "in the current reporting year"
    if horizon == 1:
        return "one year after the observed reporting year"
    return f"{horizon} years after the observed reporting year"


def format_chat(
    tokenizer: Any,
    system_msg: str | None,
    user_msg: str | None,
    *,
    include_generation_prompt: bool,
) -> str:
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    if user_msg:
        messages.append({"role": "user", "content": user_msg})
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=include_generation_prompt,
            )
        except ValueError:
            pass

    parts = []
    if system_msg:
        parts.append(f"<<SYS>>\n{system_msg}\n<</SYS>>")
    if user_msg:
        parts.append(f"[INST] {user_msg} [/INST]")
    return "\n".join(parts)
