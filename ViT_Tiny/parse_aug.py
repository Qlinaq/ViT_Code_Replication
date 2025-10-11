import argparse
import torch

def parse_args():
    p = argparse.ArgumentParser(description="ViT-Tiny with optional SPT and full LSA (diag mask + rel-pos + learned temp) + Basic Augmentations")
    # Model
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000, help="当未提供数据路径时使用；若提供data_train，将自动以数据集类数为准")
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--attn-drop-rate", type=float, default=0.0)
    p.add_argument("--drop-path-rate", type=float, default=0.0)
    # Toggles
    p.add_argument("--spt", action="store_true", help="启用 SPT patch embedding")
    p.add_argument("--lsa", action="store_true", help="启用完整 LSA（对角mask + 2D相对位置 + 可学习温度）")
    p.add_argument("--print-model", action="store_true")
    # Data paths
    p.add_argument("--data-train", type=str, default="", help="训练集根目录（ImageFolder）")
    p.add_argument("--data-val", type=str, default="", help="验证集根目录（ImageFolder，可选）")
    p.add_argument("--num-workers", type=int, default=4)
    # Basic augmentations（基础数据增广）
    p.add_argument("--rrc-scale-min", type=float, default=0.6, help="RandomResizedCrop 最小scale")
    p.add_argument("--rrc-scale-max", type=float, default=1.0, help="RandomResizedCrop 最大scale")
    p.add_argument("--hflip", type=float, default=0.5, help="RandomHorizontalFlip 概率")
    p.add_argument("--vflip", type=float, default=0.0, help="RandomVerticalFlip 概率（某些任务不适合，默认0）")
    p.add_argument("--color-jitter", type=float, default=0.4, help="ColorJitter 强度，0表示关闭")
    p.add_argument("--random-erasing", type=float, default=0.25, help="RandomErasing 概率，0表示关闭")
    p.add_argument("--re-scale-min", type=float, default=0.02, help="RandomErasing scale最小")
    p.add_argument("--re-scale-max", type=float, default=0.2, help="RandomErasing scale最大")
    p.add_argument("--re-ratio-min", type=float, default=0.3, help="RandomErasing ratio最小")
    p.add_argument("--re-ratio-max", type=float, default=3.3, help="RandomErasing ratio最大")
    # Train
    p.add_argument("--epochs", type=int, default=0, help=">0 则进行训练；若未提供data路径仍可做玩具训练")
    p.add_argument("--iters-per-epoch", type=int, default=10, help="玩具训练每epoch迭代数（无真实数据时有效）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoothing", type=float, default=0.0, help="Label smoothing for CrossEntropyLoss (0 disables)")
    p.add_argument("--mixup", action="store_true", help="启用 Mixup")
    p.add_argument("--cutmix", action="store_true", help="启用 CutMix")
    p.add_argument("--mixup-alpha", type=float, default=0.8, help="Beta 分布参数 alpha（Mixup 强度）")
    p.add_argument("--cutmix-alpha", type=float, default=1.0, help="Beta 分布参数 alpha（CutMix 强度）")
    return p.parse_args()

