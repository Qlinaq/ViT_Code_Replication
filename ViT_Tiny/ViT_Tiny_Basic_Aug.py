#!/usr/bin/env python3
# Vision Transformer Tiny with optional SPT and full LSA (diag mask + 2D rel-pos bias + learned temperature).
# Command-line toggles for SPT/LSA to form baseline and experiment groups.
# Requirements: torch >= 1.10, torchvision >= 0.11

import math
import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mixup_cutmix import apply_mixup_cutmix
from SPT import SPT_PatchEmbed
from LSA import RelPosBias2D
from parse_aug import parse_args



# =========================
# Activation: GELU (tanh approx)
# =========================

def GELU_fn(x: torch.Tensor) -> torch.Tensor:
    # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GELUApprox(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GELU_fn(x)


# =========================
# Utils
# =========================

def trunc_normal_(tensor, mean=0.0, std=1.0):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor


# =========================
# SPT Patch Embedding
# =========================

# =========================
# Standard (non-SPT) Patch Embedding
# =========================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192, norm_layer: Optional[nn.Module] = None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"Input image size ({H}x{W}) != model expected ({self.img_size[0]}x{self.img_size[1]})."
        x = self.proj(x)                     # [B, F, H/ps, W/ps]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # [B, N, F]
        x = self.norm(x)
        return x


# =========================
# 2D Relative Position Bias (for LSA)
# =========================


# =========================
# Transformer blocks
# =========================

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELUApprox, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    多头自注意力 + 可选完整 LSA：
    - 对 patch-to-patch 做对角线mask（保留CLS）
    - 加入2D相对位置偏置
    - 可学习的每头温度（正数），作用在logits上
    """
    def __init__(
        self,
        dim,
        num_heads=3,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_lsa: bool = False,
        grid_hw: Optional[Tuple[int, int]] = None,  # (Gh, Gw)
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_lsa = use_lsa

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # LSA components
        if self.use_lsa:
            assert grid_hw is not None, "grid_hw (Gh, Gw) must be provided when use_lsa=True"
            Gh, Gw = grid_hw
            self.rel_pos = RelPosBias2D(Gh, Gw, num_heads)
            # 每头可学习温度，softplus保证为正
            self.log_temp = nn.Parameter(torch.zeros(num_heads))  # temp_h = softplus(log_temp_h)
            # 预构建patch对角mask（保留CLS交互）
            Np = Gh * Gw
            N = Np + 1  # +1 for CLS
            mask = torch.zeros(N, N, dtype=torch.bool)
            if N > 1:
                patch_N = N - 1
                mask[1:, 1:] = torch.eye(patch_N, dtype=torch.bool)
            self.register_buffer("diag_mask", mask, persistent=False)
        else:
            self.rel_pos = None
            self.log_temp = None
            self.register_buffer("diag_mask", None, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, H, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, Hd)

        # 基础 scaled dot-product
        attn_logits = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)

        if self.use_lsa:
            # 1) 相对位置偏置（仅 patch-to-patch）
            Np = self.rel_pos.rel_pos_index.shape[0]  # Gh*Gw
            assert N == Np + 1, f"With LSA, expected N = 1 + Gh*Gw. Got N={N}, Gh*Gw={Np}."
            rpb = self.rel_pos()                      # [Np, Np, H]
            rpb = rpb.permute(2, 0, 1).unsqueeze(0)   # [1, H, Np, Np]
            # 为CLS补零，pad到 [1, H, N, N]
            pad_rpb = torch.zeros((1, self.num_heads, N, N), device=x.device, dtype=rpb.dtype)
            pad_rpb[:, :, 1:, 1:] = rpb
            attn_logits = attn_logits + pad_rpb

            # 2) 每头温度
            temp = torch.nn.functional.softplus(self.log_temp).view(1, -1, 1, 1)  # [1, H, 1, 1]
            attn_logits = attn_logits / temp

            # 3) 对角mask
            attn_logits = attn_logits.masked_fill(self.diag_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=GELUApprox,
        norm_layer=nn.LayerNorm,
        use_lsa: bool = False,
        grid_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_lsa=use_lsa,
            grid_hw=grid_hw,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =========================
# Vision Transformer
# =========================

class VisionTransformer(nn.Module):
    """
    ViT-Tiny/16 可选：
      - use_spt: Shifted Patch Tokenization
      - use_lsa: Locality Self-Attention（对角mask + 2D rel-pos + 可学习温度）
      - tanh-approx GELU
    默认 Tiny 配置：embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        distilled=False,
        use_spt: bool = False,
        use_lsa: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.distilled = distilled
        self.use_spt = use_spt
        self.use_lsa = use_lsa

        # Patch embedding
        if use_spt:
            self.patch_embed = SPT_PatchEmbed(
                input_shape=self.img_size,
                patch_size=patch_size,
                in_channels=in_chans,
                num_features=embed_dim,
                norm_layer=norm_layer,
                flatten=True,
            )
            num_patches = self.patch_embed.num_patches
            grid_h = self.img_size[0] // patch_size
            grid_w = self.img_size[1] // patch_size
        else:
            self.patch_embed = PatchEmbed(
                img_size=self.img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=norm_layer,
                flatten=True,
            )
            num_patches = self.patch_embed.num_patches
            grid_h = self.patch_embed.grid_size[0]
            grid_w = self.patch_embed.grid_size[1]

        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        # Positional embedding (learnable)
        extra_tokens = 2 if distilled else 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + extra_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        # Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=GELUApprox,
                norm_layer=norm_layer,
                use_lsa=use_lsa,
                grid_hw=(grid_h, grid_w) if use_lsa else None,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Heads
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(embed_dim, num_classes) if (distilled and num_classes > 0) else None

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)
        if self.head_dist is not None:
            trunc_normal_(self.head_dist.weight, std=0.02)
            if self.head_dist.bias is not None:
                nn.init.zeros_(self.head_dist.bias)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, N, C]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)  # [B, 2+N, C]
        else:
            x = torch.cat((cls_tokens, x), dim=1)               # [B, 1+N, C]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.distilled:
            return x[:, 0], x[:, 1]
        else:
            return x[:, 0]

    def forward(self, x):
        if self.distilled:
            x_cls, x_dist = self.forward_features(x)
            x_cls = self.head(x_cls)
            x_dist = self.head_dist(x_dist)
            if self.training:
                return x_cls, x_dist
            return (x_cls + x_dist) / 2
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


# =========================
# Factory helper
# =========================

def vit_tiny_patch16_224(
    num_classes: int = 1000,
    img_size: int = 224,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    distilled: bool = False,
    use_spt: bool = False,
    use_lsa: bool = False,
):
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=nn.LayerNorm,
        distilled=distilled,
        use_spt=use_spt,
        use_lsa=use_lsa,
    )


# =========================
# 数据增广与数据加载
# =========================

def build_transforms(args):
    """
    构建训练与验证/测试变换：
      - 训练：RandomResizedCrop + RandomHorizontalFlip(+可选VerticalFlip) + ColorJitter + ToTensor + Normalize + RandomErasing
      - 验证：Resize(256/224倍) + CenterCrop + ToTensor + Normalize
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tfms = [
        transforms.RandomResizedCrop(
            args.img_size,
            scale=(args.rrc_scale_min, args.rrc_scale_max)
        ),
        transforms.RandomHorizontalFlip(p=args.hflip),
    ]
    if args.vflip > 0.0:
        train_tfms.append(transforms.RandomVerticalFlip(p=args.vflip))
    if args.color_jitter > 0.0:
        train_tfms.append(
            transforms.ColorJitter(
                brightness=args.color_jitter,
                contrast=args.color_jitter,
                saturation=args.color_jitter,
                hue=min(0.5 * args.color_jitter, 0.1),
            )
        )
    train_tfms.extend([
        transforms.ToTensor(),
        normalize,
    ])
    # RandomErasing 作用在张量上，放在 Normalize 之后
    if args.random_erasing > 0.0:
        train_tfms.append(
            transforms.RandomErasing(
                p=args.random_erasing,
                scale=(args.re_scale_min, args.re_scale_max),
                ratio=(args.re_ratio_min, args.re_ratio_max),
                value='random'
            )
        )

    train_transform = transforms.Compose(train_tfms)

    # 验证/测试：轻量确定性预处理
    resize_size = int((256 / 224) * args.img_size)
    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


def build_dataloaders(args, train_transform, val_transform):
    """
    使用 ImageFolder 构建 DataLoader。
    目录结构要求：
      data_train/
        class_a/ *.jpg
        class_b/ *.jpg
      data_val/
        class_a/ *.jpg
        class_b/ *.jpg
    """
    if not args.data_train:
        return None, None, None  # 未提供数据路径，外部回退到玩具训练

    train_set = datasets.ImageFolder(args.data_train, transform=train_transform)
    num_classes = len(train_set.classes)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if args.data_val:
        val_set = datasets.ImageFolder(args.data_val, transform=val_transform)
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader, num_classes


# =========================
# 训练与评估（带增广）
# =========================

def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率，返回 list，对应每个 k 的精度（百分比）。"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)   # [B, maxk]
        pred = pred.t()                               # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def demo_train(args):
    """
    若提供 --data-train（可选 --data-val），则用基础数据增广进行真实数据训练；
    否则回退到随机张量的玩具训练（便于API冒烟测试）。
    """
    device = torch.device(args.device)

    # 构建变换与数据
    train_transform, val_transform = build_transforms(args)
    train_loader, val_loader, data_num_classes = build_dataloaders(args, train_transform, val_transform)

    # 确定类别数
    if train_loader is not None:
        num_classes = data_num_classes
    else:
        num_classes = args.num_classes

    # 构建模型
    model = vit_tiny_patch16_224(
        num_classes=num_classes,
        img_size=args.img_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        use_spt=args.spt,
        use_lsa=args.lsa,
    ).to(device)

    if args.print_model:
        print(model)

    # 优化器与损失
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    #loss_fn = nn.CrossEntropyLoss()


    #让标签平滑处理，使得模型更健壮，避免过拟合
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    if train_loader is None:
        # ============ 玩具训练（无真实数据路径） ============
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            for it in range(args.iters_per_epoch):
                x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
                y = torch.randint(0, num_classes, (args.batch_size,), device=device)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = loss_fn(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                total_loss += loss.item()

            avg = total_loss / max(1, args.iters_per_epoch)
            print(f"[Toy] Epoch {epoch}: loss={avg:.4f}")

        # 快速推理
        model.eval()
        with torch.no_grad():
            x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
            logits = model(x)
            print("Eval logits shape:", logits.shape)
        return

    # ============ 真实数据训练（带基础增广） ============
    for epoch in range(args.epochs):
        # 训练
        model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_seen = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            images, targets_a, targets_b, lam = apply_mixup_cutmix(
            images, targets,
            use_mixup=args.mixup, use_cutmix=args.cutmix,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha
            )

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                if lam is None:
                 # 未启用 mixup/cutmix，常规损失
                    loss = loss_fn(logits, targets)
                
                # mixup/cutmix：加权损失；label smoothing 在 loss_fn 内部生效
                else:
                    loss = lam * loss_fn(logits, targets_a) + (1.0 - lam) * loss_fn(logits, targets_b)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            total_loss += loss.item() * images.size(0)
            acc1 = accuracy(logits, targets, topk=(1,))[0]
            total_correct += acc1.item() / 100.0 * images.size(0)
            total_seen += images.size(0)

        train_loss = total_loss / max(1, total_seen)
        train_top1 = 100.0 * total_correct / max(1, total_seen)

        # 验证
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0.0
            val_seen = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    logits = model(images)
                    loss = loss_fn(logits, targets)

                    val_loss_sum += loss.item() * images.size(0)
                    acc1 = accuracy(logits, targets, topk=(1,))[0]
                    val_correct += acc1.item() / 100.0 * images.size(0)
                    val_seen += images.size(0)

            val_loss = val_loss_sum / max(1, val_seen)
            val_top1 = 100.0 * val_correct / max(1, val_seen)
            print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} top1 {train_top1:.2f}% "
                  f"| val_loss {val_loss:.4f} top1 {val_top1:.2f}%")
        else:
            print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} top1 {train_top1:.2f}%")


# =========================
# CLI
# =========================


def main():
    args = parse_args()

    # 若未显式训练，也未提供数据路径：进行一次前向检查，保证构建无误
    if args.epochs <= 0 and not args.data_train:
        device = torch.device(args.device)
        model = vit_tiny_patch16_224(
            num_classes=args.num_classes,
            img_size=args.img_size,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path_rate,
            use_spt=args.spt,
            use_lsa=args.lsa,
        ).to(device)

        if args.print_model:
            print(model)

        model.eval()
        x = torch.randn(2, 3, args.img_size, args.img_size, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"Forward OK. Output shape: {tuple(y.shape)}  | SPT={args.spt} LSA={args.lsa}")
        return

    # 进入训练（真实数据或玩具训练）
    demo_train(args)


if __name__ == "__main__":
    main()