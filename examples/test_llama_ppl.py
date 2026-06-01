from __future__ import annotations

import argparse
import math

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import LlamaTokenizer
from peft import PeftModel

from genft.utils.hub import resolve_modelscope_path
from llama_utils import generate_alpaca_prompt, resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perplexity with a GenFT LLaMA adapter.")
    parser.add_argument("--model_name_or_path", default="baffo32/decapoda-research-llama-7B-hf")
    parser.add_argument("--model_source", choices=["hf", "modelscope", "local"], default="hf")
    parser.add_argument("--modelscope_cache_dir", default=None)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--adapter_subfolder", default=None)
    parser.add_argument("--data_path", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train[:1000]")
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model = resolve_model_path(args.model_name_or_path, args.model_source, args.modelscope_cache_dir)
    adapter_dir = resolve_modelscope_path(args.adapter_path, args.adapter_subfolder, args.modelscope_cache_dir)
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
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()

    dataset = load_dataset("json", data_files=args.data_path, split="train") if args.data_path.endswith((".json", ".jsonl")) else load_dataset(args.data_path, split=args.split)

    def preprocess(example):
        prompt = generate_alpaca_prompt(example)
        tokens = tokenizer(prompt, truncation=True, max_length=args.cutoff_len)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(tokenized, batch_size=args.batch_size, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="ppl"):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * tokens
            total_tokens += tokens
    print(f"perplexity={math.exp(total_loss / max(total_tokens, 1)):.4f}")


if __name__ == "__main__":
    main()
