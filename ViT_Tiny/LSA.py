import torch
from torch import nn
from torch.nn.init import trunc_normal_



class RelPosBias2D(nn.Module):
    """
    为固定patch网格 (Gh, Gw) 的每个head学习2D相对位置偏置。
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

        # 预计算 (Gh*Gw, Gh*Gw) 的索引
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
        # 返回 [Gh*Gw, Gh*Gw, num_heads]
        bias = self.rel_pos_table[self.rel_pos_index.view(-1)].view(self.Gh * self.Gw, self.Gh * self.Gw, self.num_heads)
        return bias

