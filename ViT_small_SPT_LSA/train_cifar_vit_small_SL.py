import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from ViT_smaller_SPT_LSA import vit_small_SPT_LSA

# 设备设置
device = torch.device("mps" if torch.backends.mps.is_available() else (
                     "cuda:0" if torch.cuda.is_available() else "cpu"))

# 超参数
input_shape = [32, 32]
num_classes = 10
batch_size = 128 
epochs = 20
lr = 3e-4
weight_decay = 1e-4

# 数据增强与归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# 模型
model = vit_small_SPT_LSA(input_shape=input_shape, num_classes=num_classes)
model = model.to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 学习率调度器（可选）
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 训练函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

# 验证函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)
    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

def main():
    # 训练主循环
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'ViT_small_SPT_LSA/best_vit_cifar10.pth')

    print(f"Best Val Acc: {best_acc:.2f}%")


if __name__ == '__main__':
    try:
        # 在基于 spawn 的启动方法上支持可冻结程序
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()