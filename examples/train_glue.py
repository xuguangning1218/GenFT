from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from peft import GenFTConfig, get_peft_model


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
}

TASK_TO_EVAL_SPLIT = {
    "mnli": "validation_matched",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa with GenFT on GLUE.")
    parser.add_argument("--task", default="sst2", choices=sorted(TASK_TO_KEYS))
    parser.add_argument("--model_name_or_path", default="FacebookAI/roberta-base")
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--output_dir", default="outputs/glue")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--generator_share_dim", type=int, default=24)
    parser.add_argument("--individual_features", type=int, default=2)
    parser.add_argument("--individual_init_a", default="kaiming_uniform")
    parser.add_argument("--individual_init_b", default="kaiming_uniform")
    parser.add_argument("--ratio_W0", type=float, default=0.5)
    parser.add_argument("--inner_activation", default="None")
    parser.add_argument("--outer_activation", default="None")
    parser.add_argument("--scaling", type=float, default=0.05)
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--target_modules", nargs="+", default=["query", "value"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_num_labels(task: str, dataset) -> int:
    if task == "stsb":
        return 1
    return dataset["train"].features["label"].num_classes


def build_dataloaders(task: str, tokenizer, batch_size: int):
    dataset = load_dataset("nyu-mll/glue", task)
    sentence1_key, sentence2_key = TASK_TO_KEYS[task]

    def preprocess(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    remove_columns = [name for name in dataset["train"].column_names if name != "label"]
    tokenized = dataset.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized = tokenized.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    eval_split = TASK_TO_EVAL_SPLIT.get(task, "validation")
    train_loader = DataLoader(tokenized["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    eval_loader = DataLoader(tokenized[eval_split], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
    return dataset, train_loader, eval_loader


@torch.no_grad()
def evaluate_glue(model, dataloader, task: str, device: torch.device) -> float:
    metric = evaluate.load("glue", task)
    metric_name = TASK_TO_METRIC[task]
    model.eval()
    for batch in tqdm(dataloader, desc="eval"):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        if task == "stsb":
            predictions = outputs.logits[:, 0]
        else:
            predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    return metric.compute()[metric_name]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset, train_loader, eval_loader = build_dataloaders(args.task, tokenizer, args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=get_num_labels(args.task, dataset),
        ignore_mismatched_sizes=True,
        return_dict=True,
    )

    peft_config = GenFTConfig(
        task_type="SEQ_CLS",
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
        modules_to_save=["classifier", "score"],
    )
    model = get_peft_model(model, peft_config).to(device)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir) / args.task
    best_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_metric = float("-inf")
    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        score = evaluate_glue(model, eval_loader, args.task, device)
        print(f"epoch={epoch + 1} {TASK_TO_METRIC[args.task]}={score:.4f}")
        if score > best_metric:
            best_metric = score
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"saved best model to {best_dir}")

    print(f"best {TASK_TO_METRIC[args.task]}={best_metric:.4f}")


if __name__ == "__main__":
    main()
