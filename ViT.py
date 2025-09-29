import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from collections import OrderedDict
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#1.Patch嵌入
#num_features=768 is mannually set for ViT-B/16	
class PatchEmbed(nn.Module):
    #[B, C, H, W] as input
    def __init__(self,input_shape=[224,224],patch_size=16,in_channels=3,num_features=768, norm_layer=None,flatten=True):
        super(PatchEmbed,self).__init__()
        self.num_patches=(input_shape[0]//patch_size) * (input_shape[1]//patch_size) #not / but //
        self.proj=nn.Conv2d(in_channels,num_features,kernel_size=patch_size,stride=patch_size,)#定义卷积层，
        #将输入图像 （224x224）分割成 patch_size x patch_size 的小块，同时proj到 一个高维空间num_features 维度
        #卷积后：[B, num_features, H/patch_size, W/patch_size]
        self.flatten=flatten #展平后：[B, num_features, num_patches]
        #最后归一化，link CNN and Transformer
        self.norm=norm_layer(num_features) if norm_layer else nn.Identity()
    
    def forward(self,x):
        # batch, 3, 224, 224 -> batch, 768, 14, 14
        x=self.proj(x)
        if self.flatten:
            x=x.flatten(2).transpose(1,2)  # batch, 768, 14, 14 -> batch, 768, 196 -> batch, 196, 768
        x=self.norm(x)

        return x

patch_embed = PatchEmbed(input_shape=[224, 224], patch_size=16, in_channels=3, num_features=768)
x = torch.randn(1, 3, 224, 224)
output = patch_embed(x)
print("Output shape:", output.shape)








