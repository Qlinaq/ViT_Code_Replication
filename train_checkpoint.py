import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

from ViT_smaller_SPT_LSA import vit_small  # ViT_smaller.py 必须在同目录

def get_dataloaders(dataset_name="CIFAR10", input_shape=(128, 128), batch_size=8, num_workers=2):
    train_tf = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    root = "./data"
    name = dataset_name.upper()
    if name == "CIFAR10":
        train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
        val_set   = datasets.CIFAR10(root=root, train=False, download=True, transform=val_tf)
    elif name == "CIFAR100":
        train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
        val_set   = datasets.CIFAR100(root=root, train=False, download=True, transform=val_tf)
    else:
        raise ValueError("dataset_name must be CIFAR10 or CIFAR100")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    num_classes = len(train_set.classes)
    return train_loader, val_loader, num_classes

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total, correct = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def main():
    dataset_name = "CIFAR10"      # 或 "CIFAR100"
    input_shape = (128, 128)
    batch_size = 8
    total_epochs = 13           # 总共想训练的轮数（可改）
    lr = 3e-4
    weight_decay = 0.05
    num_workers = 2

    train_loader, val_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        input_shape=input_shape,
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = vit_small(input_shape=list(input_shape), num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    os.makedirs("checkpoints_small", exist_ok=True)
    checkpoint_path = f"checkpoints_small/vit_small_{dataset_name.lower()}_best.pth"

    # ==== 断点续训部分 ====
    start_epoch = 1
    best_val_acc = 0.0
    if os.path.exists(checkpoint_path):
        print(f"==> Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 判断是老式只保存state_dict，还是新版dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            start_epoch = checkpoint.get("epoch", 1) + 1
            print(f"==> Resuming from epoch {start_epoch}, best val acc = {best_val_acc:.4f}")
        else:
            # 老版只保存模型参数
            model.load_state_dict(checkpoint)
            print("==> Loaded model weights only (optimizer/scheduler not resumed)")
    else:
        print("==> No checkpoint found, training from scratch.")

    # ==== 主训练循环 ====
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running_loss, total, correct = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        # ==== 保存模型和训练状态 ====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            }
            torch.save(state, checkpoint_path)
            print(f"  -> Saved best model and state (acc={best_val_acc:.4f})")

        scheduler.step()

if __name__ == "__main__":
    main()