# V4FinBench

V4FinBench is a public benchmark for rare-event, multi-horizon financial-distress
prediction on about 1.1M firm-year rows from roughly 200k companies in the four
Visegrád countries. This repository builds the horizon labels, defines the
chronological evaluation protocol, and runs the reference models (tabular
baselines, RealMLP, TabM, and the tabular foundation models TabPFN-3.5 and
TabICL v2).

The raw EMIS extraction is not redistributed; the pipeline starts from the
released Kaggle data (see [docs/data_provenance.md](docs/data_provenance.md)).

## License

Code: MIT. Dataset on Kaggle: CC BY 4.0.

## Data

Download `company_years.parquet` from the Kaggle dataset into `data/raw/`:

https://www.kaggle.com/datasets/sebastiantomczak10/v4-group-corporate-bankruptcy/data

Only this file is used. Horizon labels are rebuilt from it
into `data/processed/company_years_h{1..5}.parquet`.

## Protocol in brief

- Firms are identified by `emis_id`; each row is a firm-year.
- A distress event is a firm's final report, filed before 2021, that meets the
  composite distress criterion.
- Horizons `h = 1..5` are calendar years; a row from year `t` is labeled by the
  firm's status in `t + h`.
- Evaluation is walk-forward: one model per task (horizon and test year,
  2013–2020), trained only on labels known by the test year. The main track
  scores the 15 tasks whose event year is 2018–2020; the full track all 25.
- The primary metric is country-stratified ROC-AUC (distressed vs healthy firms
  of the same country and year), with firm-clustered bootstrap intervals.

Full definitions: [docs/benchmark_protocol.md](docs/benchmark_protocol.md).
Commands: [docs/reproduction.md](docs/reproduction.md).

## Repository layout

```text
configs/          baseline search spaces, foundation-model configs
docs/             protocol, reproduction, data provenance
scripts/          command-line entry points
src/v4finbench/
├── data/         schema, labels, temporal splits, preprocessing, io
├── evaluation/   metrics, thresholds, result files, bootstrap summary
└── models/
    ├── baselines.py
    └── foundation/  TabPFN and TabICL: in-context, fine-tuning, sampling
tests/
```

## Development

```bash
uv sync --extra dev
uv run pre-commit install
uv run pytest
```
