import argparse
import json
from pathlib import Path

import pandas as pd

from v4finbench.llama.formatting import format_chat
from v4finbench.llama.inference import extract_yes_no, normalize_yes_no_logprobs
from v4finbench.llama.metrics import hard_prediction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Llama QLoRA adapter.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/llama/predictions.csv"),
    )
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--system-column", default="system")
    parser.add_argument("--user-column", default="user")
    parser.add_argument("--assistant-column", default="assistant")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compute-yes-no-probs", action="store_true")
    parser.add_argument(
        "--no-4bit",
        dest="use_4bit",
        action="store_false",
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_adapter(args)


def evaluate_adapter(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = args.use_4bit and torch.cuda.is_available()
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    df = pd.read_csv(args.test_file)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    predictions = []
    generated_texts = []
    p_yes_values = []
    p_no_values = []

    for idx, row in df.iterrows():
        prompt = format_chat(
            tokenizer,
            row.get(args.system_column, ""),
            row.get(args.user_column, ""),
            include_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        predictions.append(extract_yes_no(generated) or "UNKNOWN")
        generated_texts.append(generated)

        if args.compute_yes_no_probs:
            p_yes, p_no = _yes_no_probabilities(model, tokenizer, prompt)
        else:
            p_yes, p_no = None, None
        p_yes_values.append(p_yes)
        p_no_values.append(p_no)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1:,}/{len(df):,}", flush=True)

    df["prediction"] = predictions
    df["generated_text"] = generated_texts
    if args.compute_yes_no_probs:
        df["p_yes"] = p_yes_values
        df["p_no"] = p_no_values

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    metrics = hard_prediction_metrics(
        df,
        label_col=args.assistant_column,
        prediction_col="prediction",
    )
    metrics_out = args.metrics_out or args.out.with_suffix(".metrics.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote predictions to {args.out}", flush=True)
    print(f"Wrote metrics to {metrics_out}", flush=True)


def _sequence_logprob(model, tokenizer, prompt: str, answer: str) -> float:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + answer_ids], device=model.device)
    with torch.no_grad():
        log_probs = torch.log_softmax(model(input_ids).logits, dim=-1)
    start = len(prompt_ids)
    total = 0.0
    for offset, token_id in enumerate(answer_ids):
        total += log_probs[0, start + offset - 1, token_id].item()
    return total


def _yes_no_probabilities(model, tokenizer, prompt: str) -> tuple[float, float]:
    logp_yes = _sequence_logprob(model, tokenizer, prompt, "YES")
    logp_no = _sequence_logprob(model, tokenizer, prompt, "NO")
    return normalize_yes_no_logprobs(logp_yes, logp_no)


if __name__ == "__main__":
    main()
