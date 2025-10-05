import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设备优先 MPS
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

from ViT_smaller import vit_small  # 同目录下

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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
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
    print(f"Using device: {device}")

    dataset_name = "CIFAR10"     # 改为 "CIFAR100" 可切换
    input_shape = (128, 128)     # 内存紧张可用 (96,96)
    batch_size = 8           # 不够就降到 16/8
    epochs = 20
    lr = 3e-4
    weight_decay = 0.05
    num_workers = 2

    # 数据
    train_loader, val_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        input_shape=input_shape,
        batch_size=batch_size,
        num_workers=num_workers
    )
    checkpoint_dir = "checkpoints_small"
    checkpoint_path = os.path.join(checkpoint_dir, f"vit_small_{dataset_name.lower()}_best.pth")

    # 模型
    model = vit_small(input_shape=list(input_shape), num_classes=num_classes).to(device)

    # 损失 & 优化器 & 调度器
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints_small", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints_small/vit_small_{dataset_name.lower()}_best.pth")
            print(f"  -> Saved best model (acc={best_val_acc:.4f})")

        scheduler.step()

if __name__ == "__main__":
    main()