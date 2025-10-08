import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from SPT import SPT_PatchEmbed

# 设备优先 MPS（Apple Silicon），其次 CUDA，最后 CPU
if torch.backends.mps.is_available():
    device = "mps" 
    
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------- Patch Embedding --------
class PatchEmbed(nn.Module):
    def __init__(self, input_shape=[224, 224], patch_size=16, in_channels=3, num_features=384, norm_layer=None, flatten=True):
        super().__init__()
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.proj = nn.Conv2d(in_channels, num_features, kernel_size=patch_size, stride=patch_size)
        self.flatten = flatten
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                         # [B, C, H, W] -> [B, F, H/ps, W/ps]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)     # [B, F, N] -> [B, N, F]
        x = self.norm(x)
        return x

# -------- Attention --------
class Attention(nn.Module):
    def __init__(self, num_heads=6, dim=384, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        #self.scale = head_dim ** -0.5
        self.scale = nn.Parameter(torch.ones(num_heads) * (head_dim ** -0.5))


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        #attn = (q @ k.transpose(-2, -1)) * self.scale
        # 变成每个head有独立温度
        attn = torch.einsum('bhid,bhjd->bhij', q, k)  # [B, heads, N, N]
        scale = self.scale.view(1, self.num_heads, 1, 1)
        attn = attn * scale
        # === Diagonal Masking ===
        # 不mask cls token，仅mask visual token的对角线（假设cls token在第0位）
        # 所以 cls token to cls/patch 不mask，patch token to patch token才mask
        # 假设cls token在第0个，剩下N-1个是patch token
        if N > 1:
            # patch token的数量
            patch_N = N - 1
            # mask: [N, N]，只mask掉patch-patch的主对角线
            mask = torch.zeros(N, N, device=x.device, dtype=torch.bool)
            mask[1:, 1:] = torch.eye(patch_N, dtype=torch.bool, device=x.device)
            # [1, 1, N, N]，可自动broadcast到[B, heads, N, N]
            attn = attn.masked_fill(mask, float('-inf'))


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# -------- MLP --------
def gelu_approx(x):
    return F.gelu(x, approximate='tanh')

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=(0.0, 0.0)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop[0])
        self.drop2 = nn.Dropout(drop[1])

    def forward(self, x):
        x = self.fc1(x)
        x = gelu_approx(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# -------- DropPath --------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (x.ndim - 1) * (1,)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
    random_tensor = random_tensor.floor()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# -------- Transformer Encoder Block --------
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(num_heads=num_heads, dim=dim, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=(proj_drop, proj_drop))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# -------- ViT Small --------
class ViT_Small(nn.Module):
    def __init__(self, input_shape=[224, 224], patch_size=16, in_channels=3, num_classes=10,
                 dim=384, depth=8, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.0, proj_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.new_feature_shape = [input_shape[0] // patch_size, input_shape[1] // patch_size]
        self.old_feature_shape = [224 // patch_size, 224 // patch_size]
        num_patches_old = self.old_feature_shape[0] * self.old_feature_shape[1]

        # Patch embedding
        self.patch_embed = SPT_PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_channels=in_channels,
                                      num_features=dim, norm_layer=None, flatten=True)
        self.num_patches = self.patch_embed.num_patches

        # Tokens & Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_old + 1, dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, drop_path_rate=dpr[i],
                         norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(dim)
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x)                            # [B, N, C]
        cls_tok = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tok, x], dim=1)                 # [B, 1+N, C]

        # 位置编码插值（从 224 基准到当前输入）
        cls_pe = self.pos_embed[:, :1, :]
        img_pe = self.pos_embed[:, 1:, :]
        img_pe = img_pe.view(1, *self.old_feature_shape, -1).permute(0, 3, 1, 2)   # [1, C, 14, 14]
        img_pe = F.interpolate(img_pe, size=self.new_feature_shape, mode='bicubic', align_corners=False)
        img_pe = img_pe.permute(0, 2, 3, 1).flatten(1, 2)                          # [1, new_N, C]
        pos_embed = torch.cat([cls_pe, img_pe], dim=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def vit_small(input_shape=[224, 224], num_classes=10):
    return ViT_Small(input_shape=input_shape, num_classes=num_classes).to(device)

if __name__ == "__main__":
    model = vit_small(input_shape=[128, 128], num_classes=10)
    print(model)