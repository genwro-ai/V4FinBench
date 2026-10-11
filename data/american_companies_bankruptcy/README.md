# American Companies Bankruptcy

Source: https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset

The Kaggle file is a correctly aligned copy of the dataset described by
Lombardo et al. (2022), *Machine Learning for Bankruptcy Prediction in the
American Stock Market: Dataset and Benchmarks*:
https://doi.org/10.3390/fi14080244

The original GitHub CSV currently has misaligned columns. Do not replace this
file with that version. The SHA-256 checksum of the file used here is:

`cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`

Variables used in the validation:

- `X1`: current assets
- `X4`: EBITDA
- `X10`: total assets
- `X14`: current liabilities
- `X17`: total liabilities
- `status_label`: eventual company status

For a failed company, its final firm-year observation is treated as the
one-year-ahead bankruptcy event, following the labeling procedure in the
dataset paper.

