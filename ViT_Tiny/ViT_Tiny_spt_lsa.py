#!/usr/bin/env python3
# vit_tiny_spt_lsa_cli.py
# Vision Transformer Tiny with optional SPT and full LSA (diag mask + 2D rel-pos bias + learned temperature).
# Command-line toggles for SPT/LSA to form baseline and experiment groups.
# Requirements: torch >= 1.10

import math
import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class SPT_PatchEmbed(nn.Module):
    """
    Shifted Patch Tokenization (SPT):
    - Concatenate the original image with 4 half-patch shifted variants along channel dim.
    - Then apply a strided conv with kernel=stride=patch_size.
    """
    def __init__(
        self,
        input_shape=(224, 224),
        patch_size=16,
        in_channels=3,
        num_features=192,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ):
        super().__init__()
        assert isinstance(patch_size, int), "SPT assumes square patch_size (int)."
        self.patch_size = patch_size
        self.flatten = flatten

        H, W = input_shape
        assert H % patch_size == 0 and W % patch_size == 0, "Input must be divisible by patch_size."
        self.num_patches = (H // patch_size) * (W // patch_size)

        # After SPT, channels -> in_channels * 5
        self.proj = nn.Conv2d(in_channels * 5, num_features, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def shift(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        # x: [B, C, H, W]
        shift = self.patch_size // 2
        if shift == 0:
            return x
        if direction == 'left-up':
            return F.pad(x[..., :-shift, :-shift], (shift, 0, shift, 0))
        elif direction == 'right-up':
            return F.pad(x[..., :-shift, shift:], (0, shift, shift, 0))
        elif direction == 'left-down':
            return F.pad(x[..., shift:, :-shift], (shift, 0, 0, shift))
        elif direction == 'right-down':
            return F.pad(x[..., shift:, shift:], (0, shift, 0, shift))
        else:
            raise ValueError(f"Unknown shift direction: {direction}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x_list = [x]
        for d in ['left-up', 'right-up', 'left-down', 'right-down']:
            x_list.append(self.shift(x, d))
        x_cat = torch.cat(x_list, dim=1)     # [B, 5C, H, W]
        x = self.proj(x_cat)                 # [B, F, H/ps, W/ps]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # [B, N, F]
        x = self.norm(x)
        return x


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

class RelPosBias2D(nn.Module):
    """
    Learnable 2D relative position bias per head for a fixed patch grid (Gh, Gw).
    """
    def __init__(self, Gh: int, Gw: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.Gh = Gh
        self.Gw = Gw
        self.register_parameter(
            "rel_pos_table",
            nn.Parameter(torch.zeros((2 * Gh - 1) * (2 * Gw - 1), num_heads))
        )
        trunc_normal_(self.rel_pos_table, std=0.02)

        # Precompute pair-wise index for (Gh*Gw, Gh*Gw)
        coords_h = torch.arange(Gh)
        coords_w = torch.arange(Gw)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, Gh, Gw]
        coords_flat = torch.flatten(coords, 1)                                    # [2, Gh*Gw]
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]            # [2, Gh*Gw, Gh*Gw]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()                     # [Gh*Gw, Gh*Gw, 2]
        rel_coords[:, :, 0] += Gh - 1
        rel_coords[:, :, 1] += Gw - 1
        rel_coords[:, :, 0] *= 2 * Gw - 1
        rel_pos_index = rel_coords[:, :, 0] + rel_coords[:, :, 1]                 # [Gh*Gw, Gh*Gw]
        self.register_buffer("rel_pos_index", rel_pos_index, persistent=False)

    def forward(self) -> torch.Tensor:
        # Return [Gh*Gw, Gh*Gw, num_heads]
        bias = self.rel_pos_table[self.rel_pos_index.view(-1)].view(self.Gh * self.Gw, self.Gh * self.Gw, self.num_heads)
        return bias


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
    Multi-Head Self-Attention with optional full LSA:
    - Diagonal masking on patch-to-patch (exclude CLS at index 0)
    - Learnable 2D relative position bias (added to attention logits)
    - Learnable per-head temperature (positive), applies as logits / temp_h
    """
    def __init__(
        self,
        dim,
        num_heads=3,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_lsa: bool = False,
        grid_hw: Optional[Tuple[int, int]] = None,  # (Gh, Gw) for rel-pos bias
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
            # Learnable temperature per head, positive via softplus
            self.log_temp = nn.Parameter(torch.zeros(num_heads))  # temp_h = softplus(log_temp_h) ~ 0.693 at 0
            # Prebuild diag mask buffer for patch-to-patch
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

        # Base scaled dot-product
        attn_logits = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)

        if self.use_lsa:
            # 1) Add 2D relative position bias to patch-to-patch logits
            Np = self.rel_pos.rel_pos_index.shape[0]  # Gh*Gw
            assert N == Np + 1, f"With LSA, expected N = 1 + Gh*Gw. Got N={N}, Gh*Gw={Np}."
            rpb = self.rel_pos()                      # [Np, Np, H]
            rpb = rpb.permute(2, 0, 1).unsqueeze(0)   # [1, H, Np, Np]
            # Pad for CLS at [0,0] to [1, H, N, N]
            pad_rpb = torch.zeros((1, self.num_heads, N, N), device=x.device, dtype=rpb.dtype)
            pad_rpb[:, :, 1:, 1:] = rpb
            attn_logits = attn_logits + pad_rpb

            # 2) Learned temperature per head (positive), logits /= temp_h
            temp = torch.nn.functional.softplus(self.log_temp).view(1, -1, 1, 1)  # [1, H, 1, 1]
            attn_logits = attn_logits / temp

            # 3) Diagonal masking for patch-to-patch (keep CLS interactions)
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
    ViT-Tiny/16 with options:
      - use_spt: Shifted Patch Tokenization
      - use_lsa: Locality Self-Attention (diag mask + 2D rel-pos + learned temperature)
      - tanh-approx GELU in MLPs
    Default Tiny config: embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0
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
# Minimal training demo (optional)
# =========================

def demo_train(args):
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

    # Toy data: random tensors (for API smoke test only)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for it in range(args.iters_per_epoch):
            x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
            y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            total_loss += loss.item()

        avg = total_loss / max(1, args.iters_per_epoch)
        print(f"Epoch {epoch}: loss={avg:.4f}")

    # Quick inference
    model.eval()
    with torch.no_grad():
        x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
        logits = model(x)
        print("Eval logits shape:", logits.shape)


# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="ViT-Tiny with optional SPT and full LSA (diag mask + rel-pos + learned temp)")
    # Model
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--attn-drop-rate", type=float, default=0.0)
    p.add_argument("--drop-path-rate", type=float, default=0.0)
    # Toggles
    p.add_argument("--spt", action="store_true", help="Enable SPT patch embedding")
    p.add_argument("--lsa", action="store_true", help="Enable full LSA (diag mask + 2D rel-pos + learned temperature)")
    p.add_argument("--print-model", action="store_true")
    # Train demo
    p.add_argument("--epochs", type=int, default=0, help=">0 to run a toy training loop")
    p.add_argument("--iters-per-epoch", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Build once and sanity check forward
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

    # Quick forward to verify shapes
    model.eval()
    x = torch.randn(2, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Forward OK. Output shape: {tuple(y.shape)}  | SPT={args.spt} LSA={args.lsa}")

    # Optional toy train
    if args.epochs > 0:
        demo_train(args)


if __name__ == "__main__":
    main()