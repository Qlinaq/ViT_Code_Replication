import torch
import torch.nn as nn
import torch.nn.functional as F

class SPT_PatchEmbed(nn.Module):
    def __init__(self, input_shape=[224, 224], patch_size=16, in_channels=3, num_features=384, norm_layer=None, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        # SPT后通道数变为in_channels*5
        self.proj = nn.Conv2d(in_channels * 5, num_features, kernel_size=patch_size, stride=patch_size)
        self.flatten = flatten
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def shift(self, x, direction):
        # x: [B, C, H, W]
        shift = self.patch_size // 2
        if direction == 'left-up':
            return F.pad(x[..., :-shift, :-shift], (shift, 0, shift, 0))
        elif direction == 'right-up':
            return F.pad(x[..., :-shift, shift:], (0, shift, shift, 0))
        elif direction == 'left-down':
            return F.pad(x[..., shift:, :-shift], (shift, 0, 0, shift))
        elif direction == 'right-down':
            return F.pad(x[..., shift:, shift:], (0, shift, 0, shift))
        else:
            raise ValueError

    def forward(self, x):
        # x: [B, C, H, W]
        x_list = [x]
        for d in ['left-up', 'right-up', 'left-down', 'right-down']:
            x_list.append(self.shift(x, d))  # 每个方向shift
        x_cat = torch.cat(x_list, dim=1)  # [B, 5C, H, W]
        x = self.proj(x_cat)              # [B, F, H/ps, W/ps]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, N, F]
        x = self.norm(x)
        return x