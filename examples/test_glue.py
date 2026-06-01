from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peft import PeftModel

from genft.utils.hub import resolve_modelscope_path
from train_glue import build_dataloaders, evaluate_glue, get_num_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GenFT adapter on GLUE.")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--model_name_or_path", default="FacebookAI/roberta-base")
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--adapter_path", required=True, help="Local path or modelscope://repo_id.")
    parser.add_argument("--adapter_subfolder", default=None)
    parser.add_argument("--modelscope_cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_adapter_dir(adapter_dir: str) -> None:
    path = Path(adapter_dir)
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {path}. Check ADAPTER_PATH/ADAPTER_SUBFOLDER.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing adapter_model.safetensors in {path}. Check ADAPTER_PATH/ADAPTER_SUBFOLDER.")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    modules_to_save = set(config.get("modules_to_save") or [])
    if not ({"classifier", "score"} & modules_to_save):
        print(
            "Warning: adapter_config.json does not list classifier/score in modules_to_save. "
            "For GLUE, the classification head must be saved with the adapter."
        )


def load_generator_weights(model, adapter_dir: str, device: torch.device) -> None:
    adapter_weights = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"), device=str(device))
    loaded = 0
    for key, value in adapter_weights.items():
        if ".generator." not in key:
            continue
        target = model
        parts = key.split(".")
        try:
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], torch.nn.Parameter(value.to(device), requires_grad=False))
            loaded += 1
        except AttributeError:
            continue
    if loaded:
        print(f"Loaded {loaded} shared GenFT generator tensors from adapter weights.")


def load_logged_best_metric(adapter_dir: str) -> float | None:
    path = Path(adapter_dir)
    log_candidates = [
        path / "model.log",
        path.parent / "model.log",
        path.parent / "training.log",
    ]
    pattern = re.compile(r"Final best [^:]+:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    for log_path in log_candidates:
        if not log_path.exists():
            continue
        best = None
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    best = float(match.group(1))
        if best is not None:
            return best
    return None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    adapter_dir = resolve_modelscope_path(args.adapter_path, args.adapter_subfolder, args.modelscope_cache_dir)
    validate_adapter_dir(adapter_dir)
    tokenizer_name = args.tokenizer_name_or_path or adapter_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset, _, eval_loader = build_dataloaders(args.task, tokenizer, args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=get_num_labels(args.task, dataset),
        ignore_mismatched_sizes=True,
        return_dict=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False).to(device)
    load_generator_weights(model, adapter_dir, device)
    score = evaluate_glue(model, eval_loader, args.task, device)
    print(f"Real-time test result: {score:.4f}")
    logged_best = load_logged_best_metric(adapter_dir)
    if logged_best is not None:
        print(f"Loaded logged best result: {logged_best:.4f}")


if __name__ == "__main__":
    main()
