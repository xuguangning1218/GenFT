from __future__ import annotations

from pathlib import Path


def resolve_model_path(model_name_or_path: str, model_source: str, cache_dir: str | None = None) -> str:
    if model_source == "local":
        return model_name_or_path
    if model_source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise ImportError("Install modelscope or use --model_source hf/local.") from exc
        return str(Path(snapshot_download(model_name_or_path, cache_dir=cache_dir)))
    return model_name_or_path


def generate_alpaca_prompt(example: dict, train_on_output: bool = True) -> str:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "") if train_on_output else ""
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )

