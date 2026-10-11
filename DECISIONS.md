# ACIIDS branch: decisions and changes

This note summarizes what changed on the `aciids` branch relative to the
preprint code, and why. Full definitions are in
[docs/benchmark_protocol.md](docs/benchmark_protocol.md).

## Scope

- The Llama-3 QLoRA experiment was removed (code, configs, scripts, Slurm jobs,
  optional dependencies).
- The grouped 5-fold cross-validation was removed; only the chronological
  walk-forward evaluation remains.
- Horizon `h = 0` (current-year distress) was dropped; horizons are `h = 1..5`.
- TabPFN is out of scope for now (to be revisited with TabPFN 3.5, TabICL and
  proper tuning). Its code is kept but was not re-run.
- Slurm and bash reproduction scripts and the Kaggle download script were
  removed; Athena scripts will be written later.
- The country-transfer experiment is out of scope for now.

## Labels

**Firm identifier.** Firms are keyed by `emis_id`. Grouping by `company` name
merged 282 groups of distinct firms with the same name and produced 1,046
duplicate firm-year rows; `emis_id` has none. `emis_id` must stay in the
released files.

**Gap bug (fixed).** The old labels located "the report `h` years before the
final one" by counting rows, not years. 83,124 of 203,900 firms have gaps in
their reporting, so many positives landed on the wrong year:

| h | Misdated positives in the old Kaggle labels |
| --- | --- |
| 1 | 518 / 3,054 (17%) |
| 3 | 748 / 1,896 (39%) |
| 5 | 547 / 1,154 (47%) |

Horizons are now calendar years: a row from year `t` is labeled by the firm's
status in year `s = t + h`.

**Event.** Unchanged: the firm's final report, filed before 2021, has negative
equity, negative EBITDA and a current ratio of at most 0.6.

**Label.** As in the preprint, a row is positive iff the event is in year `s`
("distress exactly `h` years ahead"). An alternative "within `h` years" label
was considered and dropped.

**Dropped rows.**
- `s > 2021` (beyond the panel);
- `s = 2021` when the firm's 2021 report is distressed (it may or may not be
  final);
- rows at or after the event (other than the positive itself);
- **censored rows**, where the firm stopped reporting without distress before
  `s`. Previously these were negatives; they are now always dropped. This
  removes 9–21% of the labelable rows (93.5k at `h = 1`, up to 139k at `h = 4`); about half of all
  firms leave the panel while healthy, which looks like coverage churn rather
  than real outcomes.

**Label observation year.** Each row stores `label_observed_year`, the first
year its label was knowable (the event year for positives; for negatives, `s`
if the firm filed a non-distressed report for `s`, otherwise its next report
after `s`). The split uses it to keep not-yet-known labels out of training.

**Files.** `company_years_h{h}.parquet` with `h` the actual horizon (the old
Kaggle files were shifted by one). The repository is the source of truth; the
Kaggle dataset will be re-released from it. `verify_labels.py` was removed.

Resulting sizes:

| h | Rows | Positives | Positive rate |
| --- | --- | --- | --- |
| 1 | 902,171 | 2,544 | 0.28% |
| 2 | 771,798 | 1,933 | 0.25% |
| 3 | 656,253 | 1,619 | 0.25% |
| 4 | 561,694 | 1,330 | 0.24% |
| 5 | 475,221 | 1,198 | 0.25% |

## Features

Not used as features: identifiers and free text, three degenerate ratios,
`year`, `label_observed_year`, **`operational_status`** and
**`number_of_employees`**. Both are firm-level snapshots from data extraction
time, i.e. look-ahead leakage under a chronological protocol:
`operational_status` changes for only 154 firms and status 0 has a 5.6× higher
positive rate; `number_of_employees` changes for only 4,128 of 203,900 firms and
its band 0 has a 4× higher positive rate. `legal_form` and `state` are also
nearly constant but show no strong link to the label and are kept; the
`*_Sector` features are computed per year. Median imputation and standardization fit on the
training rows are kept for now.

## Time splits

For each horizon `h` and test year `Y` (2013–2020):

- test: all rows from year `Y`;
- pool: earlier rows with `t + h <= Y` and `label_observed_year <= Y`;
- validation: the two most recent pool years;
- train: all earlier pool rows (expanding window from 2006).

One model is trained per `(h, Y)`; parameters are fit on training years only,
following Gu, Kelly and Xiu (2020). Splits with fewer than 30 test, 30 training
or 15 validation positives are skipped. Events end in 2020, so horizon `h` is
testable up to `2020 - h`, giving 25 cells. Splits are computed on the fly from
the label files, so they cannot go stale. Test years start in 2013 because the
2012 cells had very small training sets.

## Evaluation

- Primary table: test years 2013–2015, where every horizon is available
  (`--all-years` for every year).
- Ranking metrics (ROC-AUC, PR-AUC, recall at 1% and 0.1% FPR) are computed per
  test year and averaged; F1, precision and recall come from counts summed over
  test years, each year using its validation threshold (Forman and Scholz,
  2010). Pooling scores from separately trained yearly models would penalize
  models whose score scale shifts between years.
- Uncertainty: firm-clustered bootstrap (firms resampled with all their test
  rows), 95% intervals, and paired differences between models on the same
  resamples.

## Baselines and tuning

- Models: logistic regression, XGBoost, LightGBM, CatBoost, **RealMLP**
  (priority) and **TabM**, the last two via `pytabkit`. RealMLP is tuned over
  its published search space (Holzmüller et al., 2024), on Athena GPUs; TabM
  runs with its tuned defaults. The sklearn MLP and random forest were removed.
- Compute: logistic regression and the GBDTs fit locally (about 7–8 hours for
  all 25 cells with 40 trials); RealMLP tuning needs Athena.
- The old fixed grids (100–200 trees, depth and learning rate only) were poorly
  shaped: on a diagnostic split every model preferred the lowest-capacity corner
  and validation PR-AUC peaked after 20–100 trees, i.e. the issue is overfitting
  with few positives, not too few trees.
- New tuning: 40 Optuna (TPE) trials per model and split over
  regularization-focused search spaces (`configs/baselines/standard.yaml`),
  early stopping on the validation years (up to 5,000 trees, patience 200).
- Selection metric: **validation PR-AUC**. PR-AUC emphasizes the top of the
  ranking, which is how an early-warning list is used (Saito and Rehmsmeier,
  2015; McDermott et al., 2024, who recommend PR-AUC for this kind of top-heavy
  intervention setting). The decision threshold is then chosen by validation F1.
  Both ROC-AUC and PR-AUC are reported.
- RealMLP and TabM select their best epoch on validation cross-entropy, without
  label smoothing, as the RealMLP authors recommend for AUC-type targets.
- All models use a single seed (42), as the earlier MLP baseline did.
- Each (model, horizon, test year) writes its own score and metrics files
  (with timings and hardware); finished cells are skipped, so jobs can run in
  parallel and be resubmitted.
- The evaluation also reports metrics separately for firms seen and not seen in
  training.

## Open items

- Re-release the Kaggle dataset from this repository.
- Athena job scripts for the full grid.
- Revisit imputation and scaling (tree models and TabPFN handle missing values
  natively).
- Update `CITATION.cff` once the ACIIDS title is fixed.
- On macOS, run neural models in a separate process from XGBoost/LightGBM:
  torch and Homebrew's `libomp` clash (not an issue on Linux).

## References

- Gu, Kelly, Xiu (2020). Empirical Asset Pricing via Machine Learning. RFS.
- Forman, Scholz (2010). Apples-to-Apples in Cross-Validation Studies. SIGKDD
  Explorations.
- Saito, Rehmsmeier (2015). The Precision-Recall Plot Is More Informative than
  the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS
  ONE.
- McDermott et al. (2024). A Closer Look at AUROC and AUPRC under Class
  Imbalance. NeurIPS.
- Holzmüller, Grinsztajn, Steinwart (2024). Better by Default: Strong Pre-Tuned
  MLPs and Boosted Trees on Tabular Data. NeurIPS.
- Gorishniy et al. (2025). TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling. ICLR.
