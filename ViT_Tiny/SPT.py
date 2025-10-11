import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SPT_PatchEmbed(nn.Module):
    """
    Shifted Patch Tokenization (SPT):
    - 将原图与四个半个patch位移的图像在通道维拼接。
    - 之后用 kernel=stride=patch_size 的卷积投影到patch特征。
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

        # SPT 后通道变为 in_channels * 5
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