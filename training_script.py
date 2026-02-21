import argparse
import json
import os

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from src.data_processors import DataProcessor, DataProcessorMode


def parse_target_modules(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        modules = [str(x).strip() for x in raw_value if str(x).strip()]
        return modules or None
    modules = [x.strip() for x in str(raw_value).split(",") if x.strip()]
    return modules or None


def resolve_torch_dtype(dtype_name):
    if dtype_name == "auto":
        return None
    return getattr(torch, dtype_name)


def resolve_output_path(args):
    output_path = args.output_path if args.output_path else args.lora_path
    if not output_path:
        raise ValueError("Set --output_path (or legacy --lora_path).")
    return output_path


def resolve_save_steps(train_hf, args):
    if args.save_steps is not None and args.save_steps > 0:
        return args.save_steps
    estimated = int(len(train_hf) / max(1, args.training_batch_size))
    return max(1, estimated)


def load_train_model(args):
    load_in_8bit = args.load_in_8bit
    if load_in_8bit is None:
        load_in_8bit = args.sft_type == "lora"

    if args.sft_type == "full" and load_in_8bit:
        raise ValueError("Full SFT cannot use 8-bit loading. Set --no-load_in_8bit.")

    model_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    else:
        torch_dtype = resolve_torch_dtype(args.torch_dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)
    return model


def train(train_hf, tokenizer, args):
    output_path = resolve_output_path(args)
    save_steps = resolve_save_steps(train_hf, args)

    model = load_train_model(args)

    peft_config = None
    if args.sft_type == "lora":
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=parse_target_modules(args.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_arguments = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.training_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="none",
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(f"Saving every {save_steps} steps")
    print(f"SFT type: {args.sft_type}")

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_hf,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "tokenizer": tokenizer,
        "args": training_arguments,
        "packing": args.packing,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)
    print("Training started...")
    trainer.train()
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if args.merge_weights:
        if args.sft_type != "lora":
            print("Ignoring --merge_weights for Full SFT (no LoRA adapters to merge).")
        else:
            merged_model = trainer.model.merge_and_unload()
            merged_dir = os.path.join(output_path, "merged_model")
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"Merged model written to: {merged_dir}")
    return trainer.model


def get_training_set(args, eos_token):
    if args.dcot:
        mode = DataProcessorMode.DCOT
    elif args.cot:
        mode = DataProcessorMode.COT
    else:
        raise ValueError("Need to set one mode: --dcot or --cot")

    dataset_processor = DataProcessor(
        args.train_path,
        mode=mode,
        eos=eos_token,
        epochs=args.dataset_repeats,
        seed=args.seed,
        chat_format=args.chat_format,
        max_samples=args.max_samples,
    )
    return dataset_processor.get_hf_dataset()


def maybe_load_config(config_file):
    if config_file is None:
        return {}
    with open(config_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_file}")
    return payload


def apply_config_values(args, defaults, config_payload):
    if not config_payload:
        return args

    # Direct key mapping: config key -> argparse attribute
    key_map = {
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "warmup_ratio": "warmup_ratio",
        "logging_steps": "logging_steps",
        "save_steps": "save_steps",
        "optim": "optim",
        "bf16": "bf16",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "lora_target_modules": "lora_target_modules",
        "lr_scheduler_type": "lr_scheduler_type",
        "max_grad_norm": "max_grad_norm",
        "seed": "seed",
        "max_samples": "max_samples",
        "pad_to_max": "pad_to_max",
    }

    # Alias key mapping: config key -> argparse attribute
    alias_map = {
        "model_name": "base_model_path",
        "tokenizer_name": "tokenizer_name",
        "max_length": "max_seq_length",
        "num_epochs": "epochs",
        "batch_size": "training_batch_size",
        "grad_accum": "gradient_accumulation_steps",
    }

    for cfg_key, arg_key in key_map.items():
        if cfg_key not in config_payload:
            continue
        if getattr(args, arg_key) == getattr(defaults, arg_key):
            setattr(args, arg_key, config_payload[cfg_key])

    for cfg_key, arg_key in alias_map.items():
        if cfg_key not in config_payload:
            continue
        if getattr(args, arg_key) == getattr(defaults, arg_key):
            setattr(args, arg_key, config_payload[cfg_key])

    if "use_lora" in config_payload:
        use_lora_cfg = config_payload["use_lora"]
        if use_lora_cfg is not None and args.sft_type == defaults.sft_type:
            args.sft_type = "lora" if bool(use_lora_cfg) else "full"

    return args


def finalize_args(args, defaults):
    # CLI aliases (if set) override canonical defaults.
    if args.model_name is not None and args.base_model_path == defaults.base_model_path:
        args.base_model_path = args.model_name
    if args.tokenizer_name is None:
        args.tokenizer_name = args.base_model_path
    if args.max_length is not None and args.max_seq_length == defaults.max_seq_length:
        args.max_seq_length = args.max_length
    if args.num_epochs is not None and args.epochs == defaults.epochs:
        args.epochs = args.num_epochs
    if args.batch_size is not None and args.training_batch_size == defaults.training_batch_size:
        args.training_batch_size = args.batch_size
    if args.grad_accum is not None and args.gradient_accumulation_steps == defaults.gradient_accumulation_steps:
        args.gradient_accumulation_steps = args.grad_accum
    if args.use_lora is not None:
        args.sft_type = "lora" if args.use_lora else "full"

    # Normalize data types.
    args.epochs = float(args.epochs)
    args.dataset_repeats = int(args.dataset_repeats)
    if args.max_samples is not None:
        args.max_samples = int(args.max_samples)

    return args


def parse_args():
    parser = argparse.ArgumentParser()

    # External config
    parser.add_argument(
        "--config_file",
        type=str,
        help="Optional JSON config file. Keys like model_name/max_length/use_lora are supported.",
    )

    # Data
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--train", action="store_true")

    # Model / output
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--model_name", type=str, help="Alias of --base_model_path.")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--output_path", type=str, help="Output directory for LoRA or full-SFT model.")
    parser.add_argument("--lora_path", type=str, help="Legacy alias for --output_path.")
    parser.add_argument("--sft_type", choices=["lora", "full"], default="lora")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--merge_weights", action="store_true")

    # Method mode (unchanged)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--dcot", action="store_true", help="Divergent CoT")
    parser.add_argument(
        "--chat_format",
        type=str,
        help="Options: llama_chat_simple, llama_chat_v2, llama_cot_chat, None",
    )

    # Core training knobs
    parser.add_argument("--training_batch_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None, help="Alias of --training_batch_size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=float, default=None, help="Alias of --epochs.")
    parser.add_argument(
        "--dataset_repeats",
        type=int,
        default=1,
        help="How many times to repeat dataset rows before Trainer epochs.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=None, help="Alias of --gradient_accumulation_steps.")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--group_by_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=None, help="Alias of --max_seq_length.")
    parser.add_argument("--pad_to_max", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)

    # Precision / model loading
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load_in_8bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)

    # LoRA knobs (used only when sft_type=lora)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target modules.",
    )

    # Logging / saving
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_strategy", choices=["no", "steps", "epoch"], default="no")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, help="Number of chains to generate for eval")

    args = parser.parse_args()
    defaults = parser.parse_args([])
    config_payload = maybe_load_config(args.config_file)
    args = apply_config_values(args, defaults, config_payload)
    args = finalize_args(args, defaults)
    return args


if __name__ == "__main__":
    print("Starting")
    args = parse_args()

    if args.train and not args.train_path:
        raise ValueError("Set --train_path when using --train.")
    if args.cot and args.dcot:
        raise ValueError("Use only one of --cot or --dcot.")
    if not args.cot and not args.dcot:
        raise ValueError("Set one mode: --cot or --dcot.")
    if args.base_model_path is None:
        raise ValueError("Set --base_model_path (or --model_name / config model_name).")
    if args.bf16 and args.fp16:
        raise ValueError("Use only one of --bf16 or --fp16.")
    if args.pad_to_max:
        print("Note: --pad_to_max is accepted for config compatibility and ignored by this SFTTrainer pipeline.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=args.trust_remote_code
    )
    # Llama/Phi models often do not define a pad token.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    if args.train:
        train_hf = get_training_set(args, tokenizer.eos_token)
        train(train_hf, tokenizer, args)
