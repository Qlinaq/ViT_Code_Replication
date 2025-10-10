#!/usr/bin/env python3
import os
import csv
import math
import time
import copy
import random
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# ========= Utils =========

def load_vit_factory(module_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("vit_spt_lsa", module_path)
    vit_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vit_mod)
    if not hasattr(vit_mod, "vit_tiny_patch16_224"):
        raise RuntimeError("Module does not expose vit_tiny_patch16_224")
    return vit_mod.vit_tiny_patch16_224

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def pick_device(requested: str = "") -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ========= Metrics =========

def top1_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def f1_macro(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    # 手写宏平均F1，避免外部依赖
    f1_sum = 0.0
    eps = 1e-9
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_sum += f1
    return f1_sum / num_classes

# ========= Data =========

def get_data_loaders(img_size: int, num_classes: int, batch_size: int,
                     train_subset_per_class: int, val_subset_total: int,
                     num_workers: int = 2, root: str = "/content/data") -> Tuple[DataLoader, DataLoader, int, int]:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    os.makedirs(root, exist_ok=True)
    train_set_full = tv.datasets.CIFAR100(root=root, train=True, transform=train_tf, download=True)
    val_set_full   = tv.datasets.CIFAR100(root=root, train=False, transform=val_tf, download=True)

    # 训练子集：每类取固定数量（放大过拟合倾向）
    targets = torch.tensor(train_set_full.targets)
    indices = []
    for c in range(num_classes):
        idx_c = (targets == c).nonzero(as_tuple=False).view(-1).tolist()
        random.shuffle(idx_c)
        take = min(train_subset_per_class, len(idx_c))
        indices.extend(idx_c[:take])
    random.shuffle(indices)
    train_set = Subset(train_set_full, indices)

    # 验证子集
    val_indices = list(range(len(val_set_full)))
    random.shuffle(val_indices)
    val_indices = val_indices[:val_subset_total]
    val_set = Subset(val_set_full, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(train_set), len(val_set)

# ========= Train/Eval =========

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    acc_sum, f1_sum = 0.0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        bsz = x.size(0)
        loss_sum += loss.item() * bsz
        acc_sum += top1_acc(logits, y) * bsz
        f1_sum += f1_macro(logits, y, num_classes) * bsz
        n += bsz
    return loss_sum / n, acc_sum / n, f1_sum / n

def train_one_epoch(model, loader, device, optimizer, scaler=None, num_classes: int = 100,
                    log_batches: bool = False, batch_writer=None, exp_name: str = "", epoch: int = 0):
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_sum, acc_sum, f1_sum, n = 0.0, 0.0, 0.0, 0
    use_amp = scaler is not None
    amp_device = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")

    for it, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and amp_device != "cpu":
            with torch.amp.autocast(device_type=amp_device):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

        bsz = x.size(0)
        loss_sum += loss.item() * bsz
        acc = top1_acc(logits, y)
        f1 = f1_macro(logits, y, num_classes)
        acc_sum += acc * bsz
        f1_sum += f1 * bsz
        n += bsz

        if log_batches and batch_writer is not None:
            batch_writer.writerow({
                "experiment": exp_name,
                "epoch": epoch,
                "batch_index": it,
                "batch_size": bsz,
                "batch_loss": float(loss.item()),
                "batch_top1": float(acc),
                "batch_f1": float(f1),
            })

    return loss_sum / n, acc_sum / n, f1_sum / n

# ========= Main =========

def main():
    ap = argparse.ArgumentParser(description="ViT-Tiny SPT/LSA single-run with CSV logging (overfitting test).")
    # Model file and toggles
    ap.add_argument("--model-path", type=str, default="/content/ViT_Tiny_spt_lsa.py", help="Path to ViT factory file")
    ap.add_argument("--spt", action="store_true", help="Enable SPT")
    ap.add_argument("--lsa", action="store_true", help="Enable LSA")
    # Basic training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--drop-path-rate", type=float, default=0.0)  # 放大过拟合：默认0
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    # Subset to induce overfitting
    ap.add_argument("--train-subset-per-class", type=int, default=20)  # 放大过拟合：默认20
    ap.add_argument("--val-subset-total", type=int, default=5000)
    # Logging
    ap.add_argument("--out-dir", type=str, default="/content/exp_logs")
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--log-train-batches", action="store_true", help="Also log per-batch metrics to CSV")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print("Device:", device)

    # Experiment name
    if args.exp_name == "":
        if args.spt and args.lsa:
            exp_name = "spt_lsa"
        elif args.spt:
            exp_name = "spt"
        elif args.lsa:
            exp_name = "lsa"
        else:
            exp_name = "baseline"
    else:
        exp_name = args.exp_name

    # Load model factory
    vit_factory = load_vit_factory(args.model_path)

    # Build model
    model = vit_factory(
        num_classes=args.num_classes,
        img_size=args.img_size,
        drop_path_rate=args.drop_path_rate,
        use_spt=args.spt,
        use_lsa=args.lsa,
    ).to(device)

    # Data
    train_loader, val_loader, train_size, val_size = get_data_loaders(
        img_size=args.img_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        train_subset_per_class=args.train_subset_per_class,
        val_subset_total=args.val_subset_total,
        num_workers=2
    )
    print(f"[{exp_name}] Train subset: {train_size} | Val subset: {val_size} | SPT={args.spt} LSA={args.lsa}")

    # Optimizer & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # CSV writers
    os.makedirs(args.out_dir, exist_ok=True)
    epoch_csv_path = os.path.join(args.out_dir, f"{exp_name}.csv")
    epoch_fieldnames = [
        "experiment", "epoch",
        "train_loss", "train_top1", "train_f1",
        "val_loss", "val_top1", "val_f1",
        "train_val_gap_top1"
    ]
    epoch_csv_f = open(epoch_csv_path, "w", newline="")
    epoch_writer = csv.DictWriter(epoch_csv_f, fieldnames=epoch_fieldnames)
    epoch_writer.writeheader()

    if args.log_train_batches:
        batch_csv_path = os.path.join(args.out_dir, f"{exp_name}_batches.csv")
        batch_fieldnames = ["experiment", "epoch", "batch_index", "batch_size", "batch_loss", "batch_top1", "batch_f1"]
        batch_csv_f = open(batch_csv_path, "w", newline="")
        batch_writer = csv.DictWriter(batch_csv_f, fieldnames=batch_fieldnames)
        batch_writer.writeheader()
    else:
        batch_writer = None
        batch_csv_f = None

    best_val = -1.0
    best_w = None

    for ep in range(1, args.epochs + 1):
        tl, ta, tf1 = train_one_epoch(
            model, train_loader, device, optimizer, scaler,
            num_classes=args.num_classes,
            log_batches=args.log_train_batches,
            batch_writer=batch_writer,
            exp_name=exp_name, epoch=ep
        )
        vl, va, vf1 = evaluate(model, val_loader, device, num_classes=args.num_classes)
        gap = float(max(0.0, ta - va))  # 过拟合倾向：越大越可能过拟合

        epoch_writer.writerow({
            "experiment": exp_name,
            "epoch": ep,
            "train_loss": float(tl),
            "train_top1": float(ta),
            "train_f1": float(tf1),
            "val_loss": float(vl),
            "val_top1": float(va),
            "val_f1": float(vf1),
            "train_val_gap_top1": gap,
        })
        epoch_csv_f.flush()
        if batch_csv_f is not None:
            batch_csv_f.flush()

        print(f"[{exp_name}] Epoch {ep:02d} | "
              f"train_loss={tl:.4f} top1={ta:.3f} f1={tf1:.3f} | "
              f"val_loss={vl:.4f} top1={va:.3f} f1={vf1:.3f} | gap={gap:.3f}")

        if va > best_val:
            best_val = va
            best_w = copy.deepcopy(model.state_dict())

    epoch_csv_f.close()
    if batch_csv_f is not None:
        batch_csv_f.close()

    # 保存最好权重（可选）
    torch.save(best_w if best_w is not None else model.state_dict(),
               os.path.join(args.out_dir, f"{exp_name}_best.pth"))
    print(f"[{exp_name}] Done. CSV saved to: {epoch_csv_path}")

if __name__ == "__main__":
    main()