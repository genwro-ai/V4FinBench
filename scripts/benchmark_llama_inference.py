"""Measure Llama inference cost on the prepared V4FinBench prompts.

The benchmark excludes model loading and prompt-file preparation from timed
inference. Each timed run includes tokenization, device transfer, generation,
and decoding. By default it performs one 20,000-sample run and five repeated
5,000-sample runs after an untimed warm-up.
"""

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from v4finbench.llama.formatting import format_chat

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B",
        help="Hugging Face model name or local checkpoint.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional PEFT adapter. Omit to benchmark the base checkpoint.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/llama/llama_h0_test.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/llama/inference_benchmark.json"),
    )
    parser.add_argument("--system-column", default="system")
    parser.add_argument("--user-column", default="user")
    parser.add_argument("--full-samples", type=int, default=20_000)
    parser.add_argument("--repeat-samples", type=int, default=5_000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument(
        "--no-4bit",
        dest="use_4bit",
        action="store_false",
        default=True,
        help="Load the model without NF4 quantization.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "full_samples": args.full_samples,
        "repeat_samples": args.repeat_samples,
        "repetitions": args.repetitions,
        "warmup_samples": args.warmup_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if args.repeat_samples > args.full_samples:
        raise ValueError("--repeat-samples cannot exceed --full-samples.")


def load_benchmark_records(
    path: Path,
    full_samples: int,
    repeat_samples: int,
    seed: int,
    system_column: str,
    user_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared Llama test file: {path}")
    records = pd.read_csv(path)
    required = {system_column, user_column}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Missing prompt columns: {sorted(missing)}")
    if len(records) < full_samples:
        raise ValueError(
            f"{path} contains {len(records):,} rows, but --full-samples "
            f"requests {full_samples:,}."
        )

    full = records.sample(n=full_samples, random_state=seed).reset_index(drop=True)
    repeated = full.sample(
        n=repeat_samples,
        random_state=seed + 1,
    ).reset_index(drop=True)
    return full, repeated, len(records)


def prepare_prompts(
    records: pd.DataFrame,
    tokenizer: Any,
    system_column: str,
    user_column: str,
) -> list[str]:
    return [
        format_chat(
            tokenizer,
            row.get(system_column, ""),
            row.get(user_column, ""),
            include_generation_prompt=True,
        )
        for _, row in records.iterrows()
    ]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, dict]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cuda_available = torch.cuda.is_available()
    use_4bit = args.use_4bit and cuda_available
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto" if cuda_available else "cpu",
        torch_dtype=torch.bfloat16 if cuda_available else torch.float32,
    )
    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    load_seconds = time.perf_counter() - load_start

    environment = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(device)
            for device in range(torch.cuda.device_count())
        ],
        "quantization": "NF4 4-bit" if use_4bit else "none",
        "device_map": {
            name: str(device)
            for name, device in getattr(model, "hf_device_map", {}).items()
        },
        "parameter_count": int(model.num_parameters()),
        "model_load_seconds": load_seconds,
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    return model, tokenizer, environment


def synchronize_devices(torch: Any) -> None:
    if not torch.cuda.is_available():
        return
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def reset_peak_memory(torch: Any) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    baseline = {}
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
        baseline[str(device)] = torch.cuda.memory_allocated(device) / GIB
    return baseline


def peak_memory(torch: Any) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        str(device): torch.cuda.max_memory_allocated(device) / GIB
        for device in range(torch.cuda.device_count())
    }


def count_generated_tokens(generated_ids: Any, eos_token_id: int | None) -> int:
    if eos_token_id is None:
        return int(generated_ids.numel())
    total = 0
    for row in generated_ids:
        eos_positions = (row == eos_token_id).nonzero(as_tuple=False)
        total += int(eos_positions[0].item() + 1) if len(eos_positions) else len(row)
    return total


def run_inference(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    label: str,
    progress_every: int,
) -> dict[str, Any]:
    import torch

    input_device = next(model.parameters()).device
    baseline_memory = reset_peak_memory(torch)
    synchronize_devices(torch)
    start = time.perf_counter()

    input_tokens = 0
    output_tokens = 0
    processed = 0
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset : offset + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        input_tokens += int(inputs["attention_mask"].sum().item())
        inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        generated_ids = generated_ids.detach().cpu()
        output_tokens += count_generated_tokens(
            generated_ids,
            tokenizer.eos_token_id,
        )
        tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        processed += len(batch)
        if progress_every and (
            processed == len(prompts)
            or processed // progress_every != (processed - len(batch)) // progress_every
        ):
            print(f"{label}: {processed:,}/{len(prompts):,}", flush=True)

    synchronize_devices(torch)
    elapsed = time.perf_counter() - start
    samples_per_second = len(prompts) / elapsed
    total_tokens = input_tokens + output_tokens
    return {
        "run": label,
        "samples": len(prompts),
        "elapsed_seconds": elapsed,
        "samples_per_second": samples_per_second,
        "milliseconds_per_sample": 1_000 / samples_per_second,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tokens_per_second": input_tokens / elapsed,
        "output_tokens_per_second": output_tokens / elapsed,
        "total_tokens_per_second": total_tokens / elapsed,
        "baseline_gpu_memory_gib": baseline_memory,
        "peak_gpu_memory_gib": peak_memory(torch),
    }


def summarize_repetitions(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "elapsed_seconds",
        "samples_per_second",
        "milliseconds_per_sample",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "total_tokens_per_second",
    ]
    summary = {"repetitions": len(runs)}
    for field in fields:
        values = [float(run[field]) for run in runs]
        summary[f"{field}_mean"] = statistics.mean(values)
        summary[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)

    full_records, repeat_records, dataset_samples = load_benchmark_records(
        args.test_file,
        args.full_samples,
        args.repeat_samples,
        args.seed,
        args.system_column,
        args.user_column,
    )
    model, tokenizer, environment = load_model_and_tokenizer(args)

    prompt_start = time.perf_counter()
    full_prompts = prepare_prompts(
        full_records,
        tokenizer,
        args.system_column,
        args.user_column,
    )
    repeat_prompts = prepare_prompts(
        repeat_records,
        tokenizer,
        args.system_column,
        args.user_column,
    )
    prompt_preparation_seconds = time.perf_counter() - prompt_start

    warmup_count = min(args.warmup_samples, len(repeat_prompts))
    print(f"Warm-up: {warmup_count:,} samples", flush=True)
    run_inference(
        model,
        tokenizer,
        repeat_prompts[:warmup_count],
        args.batch_size,
        args.max_new_tokens,
        "warmup",
        0,
    )

    print(f"Full run: {len(full_prompts):,} samples", flush=True)
    full_run = run_inference(
        model,
        tokenizer,
        full_prompts,
        args.batch_size,
        args.max_new_tokens,
        "full",
        args.progress_every,
    )

    repeated_runs = []
    for repetition in range(1, args.repetitions + 1):
        label = f"repeat_{repetition}"
        print(f"{label}: {len(repeat_prompts):,} samples", flush=True)
        repeated_runs.append(
            run_inference(
                model,
                tokenizer,
                repeat_prompts,
                args.batch_size,
                args.max_new_tokens,
                label,
                args.progress_every,
            )
        )

    estimated_dataset_seconds = dataset_samples / full_run["samples_per_second"]
    estimated_gpu_hours = (
        estimated_dataset_seconds / 3_600 if environment["cuda_available"] else None
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "test_file": str(args.test_file),
        "dataset_samples": dataset_samples,
        "full_samples": args.full_samples,
        "repeat_samples": args.repeat_samples,
        "repetitions": args.repetitions,
        "warmup_samples": warmup_count,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt_preparation_seconds": prompt_preparation_seconds,
        "environment": environment,
        "full_run": full_run,
        "repetition_summary": summarize_repetitions(repeated_runs),
        "estimated_complete_dataset_seconds": estimated_dataset_seconds,
        "estimated_complete_dataset_gpu_hours": estimated_gpu_hours,
        "repeated_runs": repeated_runs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    runs_path = args.out.with_suffix(".runs.csv")
    pd.DataFrame([full_run, *repeated_runs]).drop(
        columns=["baseline_gpu_memory_gib", "peak_gpu_memory_gib"]
    ).to_csv(runs_path, index=False)

    repeated = report["repetition_summary"]
    print(
        f"Throughput: {repeated['samples_per_second_mean']:.3f} ± "
        f"{repeated['samples_per_second_std']:.3f} samples/s",
        flush=True,
    )
    if estimated_gpu_hours is not None:
        print(
            f"Estimated complete-dataset cost: {estimated_gpu_hours:.3f} GPU-hours",
            flush=True,
        )
    else:
        print(
            f"Estimated complete-dataset cost: "
            f"{estimated_dataset_seconds / 3_600:.3f} CPU-hours",
            flush=True,
        )
    print(f"Wrote {args.out}", flush=True)
    print(f"Wrote {runs_path}", flush=True)


if __name__ == "__main__":
    main()
