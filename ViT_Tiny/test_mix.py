#!/usr/bin/env python3
# ViT_Tiny_Basic_Aug_Test.py
# Compare baseline vs. basic data augmentation; now supports Mixup and CutMix.
# Requires: torch, torchvision
# Expects: ViT_Tiny_Basic_Aug.py (with vit_tiny_patch16_224) and mixup_cutmix.py in the same directory.

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Import ViT-Tiny factory from your model file
from ViT_Tiny_Basic_Aug import vit_tiny_patch16_224
from mixup_cutmix import apply_mixup_cutmix


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Metrics
# -------------------------

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    B = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # [B, maxk]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / B))
    return res


# -------------------------
# Sampling helpers
# -------------------------

def get_labels_from_dataset(ds) -> List[int]:
    if hasattr(ds, "targets") and ds.targets is not None:
        return list(ds.targets)
    if hasattr(ds, "samples") and ds.samples is not None:
        return [s[1] for s in ds.samples]
    raise ValueError("Cannot extract labels from dataset; unsupported dataset type.")


def sample_indices_per_class(labels: List[int], num_classes: int, per_class: int, seed: int) -> List[int]:
    if per_class <= 0:
        return list(range(len(labels)))
    rng = random.Random(seed)
    by_cls = [[] for _ in range(num_classes)]
    for idx, y in enumerate(labels):
        by_cls[y].append(idx)
    idxs = []
    for c in range(num_classes):
        pool = by_cls[c]
        rng.shuffle(pool)
        take = min(per_class, len(pool))
        idxs.extend(pool[:take])
    rng.shuffle(idxs)
    return idxs


def sample_indices_uniform(labels: List[int], total: int, seed: int) -> List[int]:
    if total <= 0 or total >= len(labels):
        return list(range(len(labels)))
    rng = random.Random(seed)
    all_idx = list(range(len(labels)))
    rng.shuffle(all_idx)
    return all_idx[:total]


# -------------------------
# Transforms
# -------------------------

def build_transforms(args, preset: str):
    """
    preset:
      - 'baseline'  : train = Resize+CenterCrop+ToTensor+Normalize
      - 'basic_aug' : train = RRC + HFlip(+VFlip opt.) + ColorJitter + ToTensor+Normalize + RandomErasing
    val = Resize+CenterCrop+ToTensor+Normalize for both
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize_size = int((256 / 224) * args.img_size)

    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])

    if preset == "baseline":
        train_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif preset == "basic_aug":
        t = [
            transforms.RandomResizedCrop(args.img_size, scale=(args.rrc_scale_min, args.rrc_scale_max)),
            transforms.RandomHorizontalFlip(p=args.hflip),
        ]
        if args.vflip > 0.0:
            t.append(transforms.RandomVerticalFlip(p=args.vflip))
        if args.color_jitter > 0.0:
            t.append(
                transforms.ColorJitter(
                    brightness=args.color_jitter,
                    contrast=args.color_jitter,
                    saturation=args.color_jitter,
                    hue=min(0.5 * args.color_jitter, 0.1),
                )
            )
        t.extend([
            transforms.ToTensor(),
            normalize,
        ])
        if args.random_erasing > 0.0:
            t.append(
                transforms.RandomErasing(
                    p=args.random_erasing,
                    scale=(args.re_scale_min, args.re_scale_max),
                    ratio=(args.re_ratio_min, args.re_ratio_max),
                    value='random'
                )
            )
        train_transform = transforms.Compose(t)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    return train_transform, val_transform


# -------------------------
# Datasets / Loaders
# -------------------------

def build_datasets(args, preset: str):
    train_tfm, val_tfm = build_transforms(args, preset)

    if args.dataset in ["cifar100", "cifar10"]:
        root = args.data_root or "./data"
        if args.dataset == "cifar100":
            base_train = datasets.CIFAR100(root=root, train=True, transform=None, download=True)
            base_val = datasets.CIFAR100(root=root, train=False, transform=val_tfm, download=True)
            num_classes = 100
        else:
            base_train = datasets.CIFAR10(root=root, train=True, transform=None, download=True)
            base_val = datasets.CIFAR10(root=root, train=False, transform=val_tfm, download=True)
            num_classes = 10

        labels = get_labels_from_dataset(base_train)
        train_indices = sample_indices_per_class(labels, num_classes, args.train_per_class, args.seed)
        train_ds = Subset(
            type(base_train)(root=root, train=True, transform=train_tfm, download=True),
            train_indices
        )

        val_labels = get_labels_from_dataset(base_val)
        if args.val_samples > 0 and args.val_samples < len(val_labels):
            val_indices = sample_indices_uniform(val_labels, args.val_samples, args.seed)
            val_ds = Subset(base_val, val_indices)
        else:
            val_ds = base_val

        return train_ds, val_ds, num_classes

    elif args.dataset == "imagefolder":
        if not args.data_train:
            raise ValueError("dataset=imagefolder 需要提供 --data-train 路径")
        base_train = datasets.ImageFolder(args.data_train, transform=None)
        num_classes = len(base_train.classes)

        labels = get_labels_from_dataset(base_train)
        train_indices = sample_indices_per_class(labels, num_classes, args.train_per_class, args.seed)
        train_ds = Subset(datasets.ImageFolder(args.data_train, transform=train_tfm), train_indices)

        if args.data_val:
            base_val = datasets.ImageFolder(args.data_val, transform=val_tfm)
            val_labels = get_labels_from_dataset(base_val)
            if args.val_samples > 0 and args.val_samples < len(val_labels):
                val_indices = sample_indices_uniform(val_labels, args.val_samples, args.seed)
                val_ds = Subset(base_val, val_indices)
            else:
                val_ds = base_val
        else:
            val_ds = None

        return train_ds, val_ds, num_classes

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_loaders(args, train_ds, val_ds):
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False
        )
    return train_loader, val_loader


# -------------------------
# Train / Eval
# -------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, args):
    model.train()
    loss_sum = 0.0
    correct = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Mixup/CutMix if enabled
        images, targets_a, targets_b, lam = apply_mixup_cutmix(
            images, targets,
            use_mixup=args.mixup, use_cutmix=args.cutmix,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha
        )

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            if lam is None:
                loss = loss_fn(logits, targets)
            else:
                loss = lam * loss_fn(logits, targets_a) + (1.0 - lam) * loss_fn(logits, targets_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        loss_sum += loss.item() * images.size(0)
        # Note: training accuracy uses original targets (common practice; may be slightly underestimated under mixup/cutmix)
        acc1 = accuracy(logits, targets, topk=(1,))[0].item()
        correct += acc1 / 100.0 * images.size(0)
        seen += images.size(0)
    return loss_sum / max(1, seen), 100.0 * correct / max(1, seen)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    correct = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss_sum += loss.item() * images.size(0)
        acc1 = accuracy(logits, targets, topk=(1,))[0].item()
        correct += acc1 / 100.0 * images.size(0)
        seen += images.size(0)
    return loss_sum / max(1, seen), 100.0 * correct / max(1, seen)


def run_experiment(args, preset: str):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Data
    train_ds, val_ds, num_classes = build_datasets(args, preset)
    train_loader, val_loader = build_loaders(args, train_ds, val_ds)

    # Model: 默认关闭 SPT/LSA，DropPath=0，聚焦增广影响
    model = vit_tiny_patch16_224(
        num_classes=num_classes,
        img_size=args.img_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        use_spt=False,
        use_lsa=False,
    ).to(device)

    # Optim / Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # label smoothing 提升鲁棒性；与 mixup/cutmix 相容
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_top1 = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_top1 = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, args)
        if val_loader is not None:
            val_loss, val_top1 = evaluate(model, val_loader, loss_fn, device)
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_epoch = epoch
            dt = time.time() - t0
            print(f"[{preset}] Epoch {epoch:03d} | train_loss {tr_loss:.4f} top1 {tr_top1:.2f}% | "
                  f"val_loss {val_loss:.4f} top1 {val_top1:.2f}% | time {dt:.1f}s")
        else:
            dt = time.time() - t0
            print(f"[{preset}] Epoch {epoch:03d} | train_loss {tr_loss:.4f} top1 {tr_top1:.2f}% | time {dt:.1f}s")

    result = {
        "preset": preset,
        "best_val_top1": float(best_val_top1),
        "best_epoch": int(best_epoch),
        "final_epoch": int(args.epochs - 1),
        "config": {
            "img_size": args.img_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "drop_path_rate": args.drop_path_rate,
            "dataset": args.dataset,
            "train_per_class": args.train_per_class,
            "val_samples": args.val_samples,
            "mixup": bool(args.mixup),
            "mixup_alpha": float(args.mixup_alpha),
            "cutmix": bool(args.cutmix),
            "cutmix_alpha": float(args.cutmix_alpha),
            "smoothing": float(args.smoothing),
        }
    }
    return result


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser("ViT-Tiny Baseline vs Basic Augment Tester (with Mixup/CutMix)")
    # Data
    p.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10", "imagefolder"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--data-train", type=str, default="", help="for imagefolder")
    p.add_argument("--data-val", type=str, default="", help="for imagefolder")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--train-per-class", type=int, default=0, help="<=0 uses all; else sample N per class")
    p.add_argument("--val-samples", type=int, default=0, help="<=0 uses all; else sample total val samples")
    p.add_argument("--workers", type=int, default=4)
    # Model (keep simple; spt/lsa off by default inside run_experiment)
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--attn-drop-rate", type=float, default=0.0)
    p.add_argument("--drop-path-rate", type=float, default=0.0)
    # Optim
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    # Presets
    p.add_argument("--preset", type=str, default="baseline", choices=["baseline", "basic_aug"])
    p.add_argument("--run-both", action="store_true", help="run baseline and basic_aug sequentially")
    # Basic aug hyperparams (only used when preset=basic_aug)
    p.add_argument("--rrc-scale-min", type=float, default=0.6)
    p.add_argument("--rrc-scale-max", type=float, default=1.0)
    p.add_argument("--hflip", type=float, default=0.5)
    p.add_argument("--vflip", type=float, default=0.0)
    p.add_argument("--color-jitter", type=float, default=0.4)
    p.add_argument("--random-erasing", type=float, default=0.25)
    p.add_argument("--re-scale-min", type=float, default=0.02)
    p.add_argument("--re-scale-max", type=float, default=0.2)
    p.add_argument("--re-ratio-min", type=float, default=0.3)
    p.add_argument("--re-ratio-max", type=float, default=3.3)
    # Mixup / CutMix
    p.add_argument("--mixup", action="store_true", help="enable Mixup augmentation")
    p.add_argument("--mixup-alpha", type=float, default=0.8, help="Beta(alpha, alpha) for Mixup")
    p.add_argument("--cutmix", action="store_true", help="enable CutMix augmentation")
    p.add_argument("--cutmix-alpha", type=float, default=1.0, help="Beta(alpha, alpha) for CutMix")
    # Loss
    p.add_argument("--smoothing", type=float, default=0.0, help="label smoothing for CrossEntropyLoss (0 disables)")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--out-dir", type=str, default="./runs_vit_tiny_basic_aug_test")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    presets = ["baseline", "basic_aug"] if args.run_both else [args.preset]
    results = []
    for p in presets:
        print(f"\n===== Running preset: {p} | mixup={args.mixup} cutmix={args.cutmix} =====\n")
        res = run_experiment(args, p)
        results.append(res)

    # Save JSON
    out_json = Path(args.out_dir) / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {out_json.resolve()}")

    # Print summary
    print("\nSummary:")
    for r in results:
        print(f"- {r['preset']}: best_val_top1={r['best_val_top1']:.2f}% @ epoch {r['best_epoch']} "
              f"(epochs={r['config']['epochs']}, img={r['config']['img_size']}, "
              f"mixup={r['config']['mixup']}, cutmix={r['config']['cutmix']})")


if __name__ == "__main__":
    main()