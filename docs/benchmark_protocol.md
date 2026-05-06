# Benchmark Protocol

The public benchmark uses the Kaggle parquet files as canonical inputs.

Fold generation uses:

```text
n_splits = 5
random_state = 42
country_col = country
group_col = company
```

Companies are assigned to folds within each country, so every observation for a company remains in the same fold. For run `fold`, validation uses `fold`, testing uses `(fold + 1) % 5`, and training uses the remaining three folds.

The horizon file mapping is:

| Paper horizon | Kaggle file |
| --- | --- |
| `h=0` | `company_years_h1.parquet` |
| `h=1` | `company_years_h2.parquet` |
| `h=2` | `company_years_h3.parquet` |
| `h=3` | `company_years_h4.parquet` |
| `h=4` | `company_years_h5.parquet` |
| `h=5` | `company_years_h6.parquet` |

The composite distress label is assigned from a company's final observed report when all three conditions hold:

```text
Equity/total_assets < 0
EBITDA/total_assets < 0
Current_assets/short_term_liabilities <= 0.6
```

For future horizons, the final `h` reports are removed and the remaining latest report is labeled positive. Companies with a final report in 2021 are treated as unresolved because the public observation window ends in 2021; their 2021 report is removed and retained earlier reports are labeled negative.

Generated fold directories include `metadata.json` so users can compare row
counts, class counts, and split sizes across environments without loading every index file manually.

Preprocessing is fit separately inside each train/validation/test rotation:

```text
1. Drop non-feature identifiers and excluded released-schema fields.
2. Compute numeric medians on the training split only.
3. Impute train, validation, and test numeric features with training medians.
4. Fit standardization on the imputed training split only.
5. Transform validation and test with the training-fitted scaler.
```

The default released-schema drop list is:

```text
company
industry
link
num
emis_id
sector_2
sector_3
sector_4
Revenue/employee
Fixed_assets/employee
EBITDA/cash_flow
```

Decision thresholds are calibrated on the validation split by maximizing F1 on the precision-recall curve, then applied unchanged to the test split.

Sampling strategies used for TabPFN context construction are implemented under `src/v4finbench/sampling/`:

```text
none
oversample
undersample
prototype_undersampling
```

For random undersampling and prototype undersampling, the default target
minority-to-majority ratio is `0.3`, matching the paper. Prototype
undersampling clusters majority-class training observations and keeps the real observations closest to the cluster centers.
