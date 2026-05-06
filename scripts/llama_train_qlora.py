import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from v4finbench.llama.formatting import format_chat


@dataclass(frozen=True)
class QLoRAConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    train_file: Path = Path("data/llama/llama_h0_train.csv")
    output_dir: Path = Path("results/generated/llama/h0_adapter")
    system_column: str = "system"
    user_column: str = "user"
    assistant_column: str = "assistant"
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    val_split: float = 0.02
    seed: int = 42
    use_bf16: bool = True
    use_4bit: bool = True
    device_map: str = "auto"
    optim: str = "paged_adamw_8bit"
    target_modules: tuple[str, ...] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Llama QLoRA adapter.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument(
        "--no-4bit",
        dest="use_4bit",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--no-bf16",
        dest="use_bf16",
        action="store_false",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    train_qlora(config)


def train_qlora(config: QLoRAConfig) -> None:
    import inspect

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_format = "json" if config.train_file.suffix in {".json", ".jsonl"} else "csv"
    dataset = load_dataset(data_format, data_files=str(config.train_file))["train"]
    split = dataset.train_test_split(test_size=config.val_split, seed=config.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    def tokenize_fn(batch):
        texts = []
        for system_msg, user_msg, assistant_msg in zip(
            batch[config.system_column],
            batch[config.user_column],
            batch[config.assistant_column],
            strict=True,
        ):
            prompt = format_chat(
                tokenizer,
                system_msg,
                user_msg,
                include_generation_prompt=False,
            )
            texts.append(f"{prompt}\n{assistant_msg}")
        return tokenizer(texts, max_length=config.max_seq_length, truncation=True)

    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    use_4bit = config.use_4bit and torch.cuda.is_available()
    device_map = (
        "cpu"
        if config.device_map == "auto" and not torch.cuda.is_available()
        else config.device_map
    )
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=(
            torch.bfloat16
            if config.use_bf16 and torch.cuda.is_available()
            else torch.float32
        ),
    )
    model.gradient_checkpointing_enable()
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = list(config.target_modules or _detect_lora_modules(model))
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_kwargs = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "logging_steps": config.logging_steps,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "save_strategy": "steps",
        "fp16": bool(torch.cuda.is_available() and not config.use_bf16),
        "bf16": bool(torch.cuda.is_available() and config.use_bf16),
        "report_to": "none",
        "optim": config.optim if torch.cuda.is_available() else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "seed": config.seed,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in params:
        training_kwargs["eval_strategy"] = "steps"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))


def _detect_lora_modules(model) -> tuple[str, ...]:
    module_names = [name for name, _ in model.named_modules()]

    def has_module(suffix: str) -> bool:
        return any(name.endswith(suffix) for name in module_names)

    if all(has_module(suffix) for suffix in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    if has_module("c_attn") and has_module("c_proj"):
        return ("c_attn", "c_proj")
    raise ValueError("Could not auto-detect LoRA target modules.")


def _build_config(args: argparse.Namespace) -> QLoRAConfig:
    values = _load_config(args.config)
    overrides = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "use_4bit": args.use_4bit,
        "use_bf16": args.use_bf16,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    if args.target_modules:
        values["target_modules"] = tuple(
            item.strip() for item in args.target_modules.split(",") if item.strip()
        )
    if "train_file" in values:
        values["train_file"] = Path(values["train_file"])
    if "output_dir" in values:
        values["output_dir"] = Path(values["output_dir"])
    if isinstance(values.get("target_modules"), list):
        values["target_modules"] = tuple(values["target_modules"])
    allowed_keys = {field.name for field in fields(QLoRAConfig)}
    values = {key: value for key, value in values.items() if key in allowed_keys}
    return QLoRAConfig(**values)


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


if __name__ == "__main__":
    main()
