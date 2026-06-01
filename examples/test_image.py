from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from timm.models import create_model

from genft.image_dataloader.loader import construct_test_loader
from genft.image_dataloader.vtab import get_data
from genft.utils.hub import resolve_modelscope_path
from genft.vision import vision_transformer_genft  # noqa: F401 registers timm models
from train_image import evaluate_image


REQUIRED_CONFIG_KEYS = [
    "bias",
    "dpr",
    "generator_share_dim",
    "individual_features",
    "individual_init_a",
    "individual_init_b",
    "inner_activation",
    "outer_activation",
    "ratio_W0",
    "scaling",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ViT-GenFT checkpoint.")
    parser.add_argument("--adapter_path", required=True, help="Local path or modelscope://repo_id.")
    parser.add_argument("--adapter_subfolder", default=None)
    parser.add_argument("--modelscope_cache_dir", default=None)
    parser.add_argument("--benchmark", choices=["vtab", "fgvc"], default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--class_num", type=int, default=None)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--model", default="vit_base_patch16_224_in21k_genft")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def load_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "args.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def has_image_checkpoint(path: Path) -> bool:
    return (path / "best_model.pt").exists() or (path / "args.yaml").exists()


def resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    checkpoint_dir = Path(resolve_modelscope_path(args.adapter_path, args.adapter_subfolder, args.modelscope_cache_dir))
    if has_image_checkpoint(checkpoint_dir):
        return checkpoint_dir

    if args.adapter_subfolder is not None and not args.adapter_path.startswith("modelscope://"):
        direct_dir = Path(resolve_modelscope_path(args.adapter_path, None, args.modelscope_cache_dir))
        if has_image_checkpoint(direct_dir):
            print(
                f"Using adapter_path directly because no image checkpoint was found at {checkpoint_dir}. "
                f"Resolved checkpoint directory: {direct_dir}"
            )
            return direct_dir

    return checkpoint_dir


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_dir = resolve_checkpoint_dir(args)
    config = load_config(checkpoint_dir)
    if "benchmark" not in config and "task" in config:
        config["benchmark"] = config["task"]
    for key in ["benchmark", "dataset"]:
        if getattr(args, key) is None and key in config:
            setattr(args, key, config[key])
    if "class_num" in config:
        args.class_num = config["class_num"]
    if args.benchmark is None or args.dataset is None or args.class_num is None:
        raise ValueError("benchmark, dataset, and class_num are required unless args.yaml is present.")
    if not (checkpoint_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"Missing image checkpoint: {checkpoint_dir / 'best_model.pt'}")
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required GenFT image config keys in {checkpoint_dir / 'args.yaml'}: {missing_keys}. "
            "Copy args.yaml next to best_model.pt, or point ADAPTER_PATH/ADAPTER_SUBFOLDER to the directory that contains them."
        )

    if args.benchmark == "vtab":
        _, test_loader = get_data(args.dataset_dir, args.dataset, evaluate=True, train_aug=False, batch_size=args.batch_size)
    else:
        test_loader = construct_test_loader(args.dataset_dir, args.dataset, batch_size=args.batch_size)

    model_args = argparse.Namespace(
        **{**vars(args), **config, "benchmark": args.benchmark, "dataset": args.dataset, "class_num": args.class_num, "tuning_mode": "genft"}
    )
    model = create_model(
        args.model,
        checkpoint_path=args.model_checkpoint,
        drop_path_rate=float(config.get("dpr", 0.1)),
        tuning_mode="genft",
        my_args=model_args,
    )
    model.reset_classifier(int(args.class_num))

    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    state = model.state_dict()
    for name, value in checkpoint["model"].items():
        state[name] = value.detach().cpu() if hasattr(value, "detach") else value
    model.load_state_dict(state, strict=False)
    model.to(device)

    acc = evaluate_image(model, test_loader, args.benchmark, device)
    print(f"Real-time test result: {acc}")
    if "best_acc" in checkpoint:
        print(f"Loaded test result: {checkpoint['best_acc']}")
    if "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    elif "ep" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['ep']}")


if __name__ == "__main__":
    main()
