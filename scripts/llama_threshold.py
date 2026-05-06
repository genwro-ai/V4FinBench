import argparse
import json
from pathlib import Path

import pandas as pd

from v4finbench.llama.metrics import threshold_probability_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose the best F1 threshold from Llama p_yes scores."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--prob-column", default="p_yes")
    parser.add_argument("--label-column", default="assistant")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    thresholded, metrics = threshold_probability_predictions(
        df,
        prob_col=args.prob_column,
        label_col=args.label_column,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    thresholded.to_csv(args.out, index=False)
    metrics_out = args.metrics_out or args.out.with_suffix(".metrics.json")
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote thresholded predictions to {args.out}", flush=True)
    print(f"Wrote metrics to {metrics_out}", flush=True)


if __name__ == "__main__":
    main()
