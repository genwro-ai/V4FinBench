# Benchmark Protocol

## Observations

Each row is a firm-year: the financial statements of firm `f` (identified by
`emis_id`) for fiscal year `t`. Firms can skip reporting years, so a firm's
history may have gaps.

## Distress event

A firm has a distress event in year `E` when its final report is filed in
`E < 2021` and meets all three conditions:

```text
Equity/total_assets < 0
EBITDA/total_assets < 0
Current_assets/short_term_liabilities <= 0.6
```

Firms still reporting in 2021, the last panel year, have no observed event.

## Horizon labels

For horizon `h ∈ {1, …, 5}`, a row from year `t` has target year `s = t + h`.
Horizons are calendar years, so reporting gaps never move a label onto another
year. One file is built per horizon: `company_years_h{h}.parquet`.

A row is positive when `E == s`. Rows with `t < E < s` and rows at or after the
event (`t >= E`) are dropped. A row is negative when the firm is known not to have had its event by `s`: it
reported after `s`, or its report for `s` does not meet the distress criterion.

Rows whose outcome at `s` was never observed are dropped:

- `s > 2021`, beyond the panel;
- `s == 2021` when the firm's 2021 report meets the distress criterion, since it
  may or may not be the final report;
- censored rows, where the firm stopped reporting without distress before `s`.

## Label observation year

The label refers to year `s`, but the evidence for it can arrive later. Each
row stores `label_observed_year`, the first year in which its label was known:

| Case | `label_observed_year` |
| --- | --- |
| positive | `E` |
| negative, non-distressed report filed for `s` | `s` |
| negative otherwise (no report for `s`, or a distressed report that was not the last) | year of the firm's next report after `s` |

The walk-forward split uses this column to keep labels that were still unknown
at prediction time out of training and validation.

## Tasks and walk-forward splits

A task is a prediction origin `Y` (the end of fiscal year `Y`) and a horizon
`h`: using everything observable by the end of `Y`, predict distress in year
`Y + h` for the firms reporting in `Y`. Each task has its own split, computed
from the horizon file on demand (`v4finbench.data.splits.temporal_splits`), for
`Y` from 2013 to 2020:

- **test**: all rows from year `Y`;
- **pool**: rows from earlier years with `t + h <= Y` and
  `label_observed_year <= Y`;
- **validation**: the two most recent years in the pool;
- **train**: all earlier pool rows (an expanding window from 2006).

This follows the expanding-train, rolling-validation design of Gu, Kelly and Xiu
(2020). Model parameters are fit on training years only; validation selects
hyperparameters and the decision threshold. `--availability-margin 1` applies
the stricter cutoff `Y - 1`.

A split is skipped when it has fewer than 30 test, 30 training or 15 validation
positives. Events end in 2020, so the last usable test year for horizon `h` is
`2020 - h`, which leaves 25 tasks. `scripts/build_temporal_splits.py` prints the
grid and writes `cells.txt` and `tasks_h{h}.parquet`, which list each task's
train, validation and test rows by (`emis_id`, `year`).

## Tracks

- **Main** (15 tasks): event years `Y + h` in 2018–2020, i.e. test years
  2017–2019 for `h = 1` down to 2013–2015 for `h = 5`. Every horizon is scored
  on the same failures, and the event years 2016–2017, which the Poland 2016
  coverage gap distorts, are avoided. This is the headline evaluation.
- **Full** (25 tasks): all tasks, for performance over time and robustness.

The same firm can appear in training and test, as it would for a deployed
model. Score files flag test rows of firms absent from training
(`unseen_firm`), and the evaluation reports metrics for all, seen and unseen
firms.

## Features and preprocessing

Not used as features: identifiers and free text (`emis_id`, `company`,
`industry`, `link`, `num`, `sector_2..4`), degenerate ratios
(`Revenue/employee`, `Fixed_assets/employee`, `EBITDA/cash_flow`), `year`,
`label_observed_year`, `operational_status` and `number_of_employees`. The last
two are firm-level snapshots taken at data extraction (they change over time for
only 154 and 4,128 of 203,900 firms), so they carry information from after
every test year.

Within each split, features are imputed with training medians and standardized
with a scaler fit on the training rows.

## Metrics

Each model scores its test year. The decision threshold maximizes F1 on the
validation years and is applied unchanged to the test year. Metrics are pooled
over a track's tasks for each model and horizon:

- **ROC-AUC (primary), country-stratified**: the probability that a distressed
  firm is ranked above a healthy firm from the same country and test year,
  computed from all such pairs over the track's tasks (covariate-adjusted ROC;
  Janes and Pepe, 2008). It never compares scores of separately trained yearly
  models (Forman and Scholz, 2010) or across countries, whose share of each test
  year is an artifact of EMIS coverage.
- **Recall at 1% and 0.1% FPR**, stratified the same way: the share of
  distressed firms caught when, in each country and year, the alert threshold
  flags that fraction of healthy firms.
- **PR-AUC** (average precision) per task, averaged over tasks; the base rate is
  about 0.25–0.6%.
- **Pooled ROC-AUC** per task, averaged over tasks, for comparison with prior
  work.
- **Precision, recall, F1** from true positives, false positives and false
  negatives summed over tasks.

Uncertainty comes from a firm-clustered bootstrap: firms are resampled with
replacement (all their test rows together) and the metrics recomputed, giving
95% percentile intervals. Paired differences between models use the same
resamples. Metrics are also reported for seen and unseen firms and per country.
All compared models must cover the same tasks.

## Models and tuning

Baselines: logistic regression, XGBoost, LightGBM, CatBoost, RealMLP and TabM,
the last two through `pytabkit`. RealMLP is tuned over the search space
published with it (Holzmüller et al., 2024); TabM runs with its tuned defaults.

For every split, each model with a search space in
`configs/baselines/standard.yaml` gets 40 Optuna (TPE) trials. Gradient-boosted
models train up to 5,000 trees with early stopping on the validation years
(patience 200). Trials are compared on validation PR-AUC, which emphasizes the
top of the risk ranking where an early-warning list is acted on (Saito and
Rehmsmeier, 2015; McDermott et al., 2024). RealMLP and TabM select their best
epoch on validation cross-entropy, without label smoothing. The decision
threshold of the selected model maximizes validation F1.

## Foundation models

TabPFN-3.5 and TabICL v2 are evaluated in context, with the training set as
context: full, randomly undersampled, randomly oversampled or prototype
undersampled (minority-to-majority ratio 0.3). Fine-tuned variants train on the
full, undersampled or prototype-undersampled training set for up to 10 epochs
with early stopping on validation ROC-AUC (the metrics the libraries offer);
among learning rates 1e-5, 3e-5 and 1e-4 the run with the best validation
PR-AUC is kept, together with its best-epoch weights. If no epoch beats the
pretrained model on validation, the pretrained weights are kept. Oversampling
is not used for fine-tuning: duplicated positives would land in both the
context and query parts of a training batch. Foundation models are run on the
main track.

## Sampling

`v4finbench.models.foundation.sampling` implements `none`, `oversample`,
`undersample` and `prototype_undersampling`. Undersampling targets a
minority-to-majority ratio of 0.3. Prototype undersampling clusters the majority
class with k-means (in torch, on CUDA, Apple GPU or CPU, iterated until the
mean squared distance improves by less than 0.01%) and keeps the real
observations closest to the cluster centers. The selected rows are cached per
training set (`sampling_cache_dir`), so every model and tuning trial on a task
reuses them.
