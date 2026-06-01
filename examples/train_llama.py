from __future__ import annotations

import argparse

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer
from peft import GenFTConfig, get_peft_model, prepare_model_for_kbit_training

from llama_utils import generate_alpaca_prompt, resolve_model_path


class GenFTTrainer(transformers.Trainer):
    """Avoid passing Trainer loss kwargs into model-parallel LLaMA forwards."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruction-tune LLaMA with GenFT.")
    parser.add_argument("--model_name_or_path", default="baffo32/decapoda-research-llama-7B-hf")
    parser.add_argument("--model_source", choices=["hf", "modelscope", "local"], default="hf")
    parser.add_argument("--modelscope_cache_dir", default=None)
    parser.add_argument("--data_path", default="yahma/alpaca-cleaned")
    parser.add_argument("--output_dir", default="outputs/llama/genft-alpaca")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--train_set_size", type=int, default=-1)
    parser.add_argument("--val_set_size", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--generator_share_dim", type=int, default=4)
    parser.add_argument("--individual_features", type=int, default=4)
    parser.add_argument("--individual_init_a", default="kaiming_uniform")
    parser.add_argument("--individual_init_b", default="kaiming_uniform")
    parser.add_argument("--ratio_W0", type=float, default=1.0)
    parser.add_argument("--inner_activation", default="gelu")
    parser.add_argument("--outer_activation", default="gelu")
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--train_on_inputs", action="store_true")
    parser.add_argument("--add_eos_token", action="store_true")
    parser.add_argument("--group_by_length", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(torch.autograd, "graph") and hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    base_model = resolve_model_path(args.model_name_or_path, args.model_source, args.modelscope_cache_dir)
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    world_size = int(__import__("os").environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(__import__("os").environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
    if ddp:
        gradient_accumulation_steps //= world_size

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model, 
        use_fast=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    peft_config = GenFTConfig(
        task_type="CAUSAL_LM",
        generator_share_dim=args.generator_share_dim,
        individual_features=args.individual_features,
        individual_init_a=args.individual_init_a,
        individual_init_b=args.individual_init_b,
        ratio_W0=args.ratio_W0,
        inner_activation=args.inner_activation,
        outer_activation=args.outer_activation,
        scaling=args.scaling,
        drop=args.drop,
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.accepts_loss_kwargs = False
    model.print_trainable_parameters()

    if args.data_path.endswith((".json", ".jsonl")):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)
    if args.train_set_size > 0:
        data["train"] = data["train"].shuffle(seed=42).select(range(args.train_set_size + args.val_set_size))

    def tokenize(prompt: str, add_eos_token: bool = True):
        result = tokenizer(prompt, truncation=True, max_length=args.cutoff_len, padding=False, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < args.cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(example):
        full_prompt = generate_alpaca_prompt(example)
        tokenized = tokenize(full_prompt, add_eos_token=args.add_eos_token)
        if not args.train_on_inputs:
            user_prompt = generate_alpaca_prompt(example, train_on_output=False)
            user_tokens = tokenize(user_prompt, add_eos_token=args.add_eos_token)
            user_prompt_len = len(user_tokens["input_ids"]) - int(args.add_eos_token)
            tokenized["labels"] = [-100] * user_prompt_len + tokenized["labels"][user_prompt_len:]
        return tokenized

    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = GenFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=args.logging_steps,
            optim="adamw_torch",
            eval_strategy="steps" if val_data is not None else "no",
            save_strategy="steps",
            eval_steps=args.eval_steps if val_data is not None else None,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=val_data is not None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=args.group_by_length,
            report_to=[],
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )
    trainer.model_accepts_loss_kwargs = False
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
