# Data Provenance

V4FinBench was constructed from company-level financial statement data extracted
from EMIS, together with company metadata such as country, industry
classification, legal form, operational status and number of employees. Firms are
identified by their EMIS identifier (`emis_id`).

The raw EMIS files are not redistributed. The public pipeline starts from the
released Kaggle file `company_years.parquet`; all horizon labels are derived from
it by `scripts/build_labels.py`.

The released Kaggle dataset is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0). The code in this repository is licensed separately
under the MIT License.
