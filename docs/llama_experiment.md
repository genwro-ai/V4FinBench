# Llama-3-8B QLoRA Experiment

The Llama-3-8B QLoRA experiment is intentionally not part of the core
reproduction path in this repository yet.

The paper evaluates Llama-3-8B under a separate compute-constrained setup:

- 20,000 sampled observations per horizon,
- 80/20 train/validation split,
- up to 100,000 test observations per horizon,
- serialized company-year records,
- QLoRA fine-tuning of `meta-llama/Meta-Llama-3-8B`,
- token-probability evaluation for `YES` and `NO`.

Do not use the legacy `economic-data/scripts/llama_evaluation.py` path as the
public reproduction implementation. A future Llama reproduction module should
live separately from the tabular benchmark runners, for example under
`scripts/run_llama_qlora.py` and `src/v4finbench/models/llama.py`.
