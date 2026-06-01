from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim import AdamW
from tqdm import tqdm

from genft.image_dataloader.loader import construct_test_loader, construct_train_loader
from genft.image_dataloader.vtab import get_data
from genft.utils.image_utils import create_logger, set_seed
from genft.vision import vision_transformer_genft  # noqa: F401 registers timm models


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT-GenFT on VTAB or FGVC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", choices=["vtab", "fgvc"], default="vtab")
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--class_num", type=int, default=100)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dpr", type=float, default=0.1)
    parser.add_argument("--train_aug", type=str2bool, default=False)
    parser.add_argument("--labelsmoothing", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--cycle_decay", type=float, default=0.1)
    parser.add_argument("--model", default="vit_base_patch16_224_in21k_genft")
    parser.add_argument("--tuning_mode", default="genft")
    parser.add_argument("--bias", type=str2bool, default=False)
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--individual_init_a", default="zero")
    parser.add_argument("--individual_init_b", default="zero")
    parser.add_argument("--ratio_W0", type=float, default=1.4)
    parser.add_argument("--inner_activation", default="None")
    parser.add_argument("--outer_activation", default="leakyrelu")
    parser.add_argument("--generator_share_dim", type=int, default=32)
    parser.add_argument("--individual_features", type=int, default=5)
    return parser.parse_args()


def build_loaders(args):
    if args.benchmark == "vtab":
        return get_data(args.dataset_dir, args.dataset, evaluate=True, train_aug=args.train_aug, batch_size=args.batch_size)
    return (
        construct_train_loader(args.dataset_dir, args.dataset, batch_size=args.batch_size),
        construct_test_loader(args.dataset_dir, args.dataset, batch_size=args.batch_size),
    )


def batch_to_xy(batch, benchmark, device):
    if benchmark == "vtab":
        return batch[0].to(device), batch[1].to(device)
    return batch["image"].float().to(device), batch["label"].to(device)


@torch.no_grad()
def evaluate_image(model, dataloader, benchmark, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="eval"):
        x, y = batch_to_xy(batch, benchmark, device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def save_args(args, output_dir: Path) -> None:
    with (output_dir / "args.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = Path(args.output_dir) / args.benchmark / args.dataset / f"save-{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_args(args, exp_dir)
    logger = create_logger(str(exp_dir), "training")
    logger.info(args)

    train_loader, test_loader = build_loaders(args)
    model = create_model(
        args.model,
        checkpoint_path=args.model_checkpoint,
        drop_path_rate=args.dpr,
        tuning_mode=args.tuning_mode,
        my_args=args,
    )
    model.reset_classifier(args.class_num)
    model.to(device)

    trainable = []
    for name, param in model.named_parameters():
        if "head" in name or "genft_" in name:
            trainable.append(param)
            logger.info(name)
        else:
            param.requires_grad = False

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        warmup_t=args.warmup_epochs,
        lr_min=1e-5,
        warmup_lr_init=1e-6,
        cycle_decay=args.cycle_decay,
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=args.labelsmoothing) if args.labelsmoothing > 0 else torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        total = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            x, y = batch_to_xy(batch, args.benchmark, device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            total += y.numel()
        scheduler.step(epoch)
        logger.info("epoch=%s loss=%.6f", epoch + 1, loss_sum / max(total, 1))

        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            acc = evaluate_image(model, test_loader, args.benchmark, device)
            logger.info("epoch=%s acc=%.4f", epoch + 1, acc)
            if acc > best_acc:
                best_acc = acc
                checkpoint = {
                    "epoch": epoch + 1,
                    "model": {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad},
                    "best_acc": best_acc,
                }
                torch.save(checkpoint, exp_dir / "best_model.pt")

    logger.info("best_acc=%.4f", best_acc)


if __name__ == "__main__":
    main()

