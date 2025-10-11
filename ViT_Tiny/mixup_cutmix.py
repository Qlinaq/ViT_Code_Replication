# mixup_cutmix.py
# Utilities for Mixup and CutMix with PyTorch Tensors.
#This is for prevention of overfitting and improve generalization.

import torch

__all__ = [
    "apply_mixup_cutmix",
]

def rand_bbox(W: int, H: int, lam: torch.Tensor, device: torch.device):
    """
    生成 CutMix 的矩形框坐标 [x1, y1, x2, y2]
    lam: 混合系数（tensor 标量，0~1）
    """
    # 按照论文，cut ratio 来源于 sqrt(1 - lam)
    cut_rat = torch.sqrt(1.0 - lam)
    cut_w = (W * cut_rat).long()
    cut_h = (H * cut_rat).long()

    # 随机中心点
    cx = torch.randint(0, W, (1,), device=device).item()
    cy = torch.randint(0, H, (1,), device=device).item()

    x1 = max(cx - cut_w.item() // 2, 0)
    y1 = max(cy - cut_h.item() // 2, 0)
    x2 = min(cx + cut_w.item() // 2, W)
    y2 = min(cy + cut_h.item() // 2, H)
    return x1, y1, x2, y2

def apply_mixup_cutmix(images: torch.Tensor, targets: torch.Tensor, use_mixup: bool, use_cutmix: bool,
                       mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0):
    """
    对一个 batch 应用 Mixup 或 CutMix（任选或随机二选一）。
    输入:
      - images: FloatTensor [B, C, H, W]
      - targets: LongTensor [B]
      - use_mixup: 是否启用 Mixup
      - use_cutmix: 是否启用 CutMix
      - mixup_alpha: Beta 分布参数（>0 启用）
      - cutmix_alpha: Beta 分布参数（>0 启用）
    输出:
      - images_aug: 变换后的图像张量
      - targets_a: 第一组目标
      - targets_b: 第二组目标（被混合/替换的那部分）
      - lam: 混合系数（标量 Tensor）; 若未启用则返回 None
    说明:
      - 若两者都启用，则每个 batch 以 0.5 概率随机挑一种。
      - 若都不启用或 alpha<=0，则直接返回原图，lam=None。
    """
    device = images.device
    B, C, H, W = images.shape

    use_mixup = bool(use_mixup and mixup_alpha > 0.0)
    use_cutmix = bool(use_cutmix and cutmix_alpha > 0.0)
    if not use_mixup and not use_cutmix:
        return images, targets, None, None

    # 选择本 batch 使用哪种
    if use_mixup and use_cutmix:
        use_mixup_now = torch.rand(1, device=device).item() < 0.5
    else:
        use_mixup_now = use_mixup

    alpha = mixup_alpha if use_mixup_now else cutmix_alpha
    # 采样 lambda
    lam = torch.distributions.Beta(alpha, alpha).sample().to(device)

    # 打乱索引
    index = torch.randperm(B, device=device)
    images_shuf = images[index]
    targets_a = targets
    targets_b = targets[index]

    if use_mixup_now:
        # 像素级线性混合
        images = lam * images + (1.0 - lam) * images_shuf
        return images, targets_a, targets_b, lam
    else:
        # CutMix：矩形区域替换
        x1, y1, x2, y2 = rand_bbox(W, H, lam, device)
        images[:, :, y1:y2, x1:x2] = images_shuf[:, :, y1:y2, x1:x2]
        # 按实际面积修正 lam
        box_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - box_area / float(W * H)
        lam = torch.tensor(lam, device=device, dtype=images.dtype)
        return images, targets_a, targets_b, lam