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
        """
        前向传播函数
        :param x: 输入图像张量，形状为 (batch, channels, height, width)
        :return: 处理后的张量，形状为 (batch, num_patches, embedding_dim)
        """
        # 将输入图像通过卷积层self.proj进行分块和线性嵌入
        # 形状变化: (batch, 3, 224, 224) -> (batch, 768, 14, 14)
        x=self.proj(x)
        if self.flatten:
            # 将特征图展平并将维度转换
            # 形状变化: (batch, 768, 14, 14) -> (batch, 768, 196) -> (batch, 196, 768)
            x=x.flatten(2).transpose(1,2)
        # 对嵌入后的块进行归一化处理
        x=self.norm(x)

        return x

class Add_CLS_Token(nn.Module):
    """CLS Token 是一个特殊的向量，表示分类信息
    它会与图像的 Patch Tokens 一起输入到 Transformer Encoder 中，并在最终用于分类任务"""
    def __init__(self,embed_dim=768):
        super(Add_CLS_Token,self).__init__()
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim)) #初始化CLS Token
    def forward(self,x):
        """
        前向传播函数
        :param x: 输入张量，形状为 (batch, num_patches, embedding_dim)
        :return: 包含 CLS Token 的输出张量，形状为 (batch, num_patches+1, embedding_dim)
        """
        
        x=torch.cat([self.cls_token.expand(x.shape[0],-1,-1),x],dim=1) #在第1维拼接CLS Token])

        return x
class Add_Position_Embed(nn.Module):
    """位置编码(Position Embedding)是为了让模型能够感知输入数据中各个部分的相对或绝对位置
    在ViT中,位置编码会与图像的 Patch Tokens 一起输入到 Transformer Encoder 中"""
    def __init__(self,num_patches=196,embed_dim=768,drop_rate=0.1):
        super(Add_Position_Embed,self).__init__()
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+1,embed_dim)) #初始化位置编码
        self.droupout=nn.Dropout(p=drop_rate)         # 应用 dropout 正则化，请注意这里的 `droupout` 可能是 `dropout` 的拼写错误

    def forward(self,x):
        """
        前向传播函数
        :param x: 输入张量，形状为 (batch, num_patches+1, embedding_dim)
        :return: 添加位置编码后的输出张量，形状为 (batch, num_patches+1, embedding_dim)
        """
        x=x+self.pos_embed #将位置编码加到输入张量上
        x=self.droupout(x)

        return x

#class EncoderBlock(nn.Module):

class Multi_Head_Attention(nn.Module):
    def __init__(self,num_heads=8,dim_of_patch=768,qkv_bias=False,attn_drop_rate=0.1,proj_drop_rate=0.1):
        super(Multi_Head_Attention,self).__init__()
        self.num_heads=num_heads
        self.scale=(dim_of_patch//num_heads)**(-0.5) #缩放因子，用于缩放注意力分数，防止 softmax 函数输出过大或过小的数值
        self.big_qkv_with_W_for_heads=nn.Linear(dim_of_patch,3*dim_of_patch,bias=qkv_bias) #线性变换层，用于生成查询（Q）、键（K）、值（V）向量
        self.atten_drop=nn.Dropout(attn_drop_rate) #注意力权重的 dropout 层，用于防止过拟合
        self.proj_AfterCatResult=nn.Linear(dim_of_patch,dim_of_patch)
        self.proj_drop=nn.Dropout(proj_drop_rate)

    def forward(self,x):
        B,N,C=x.shape 
        Atten_x=self.big_qkv_with_W_for_heads(x) 
        Atten_x=Atten_x.reshape(B,N,3,self.num_heads,C//self.num_heads)
        Atten_x=Atten_x.permute(2,0,3,1,4) # (3,B,num_heads,N,C//num_heads)
        q,k,v=Atten_x[0],Atten_x[1],Atten_x[2] #每个头的查询、键、值向量


        # 计算注意力分数
        atten=(q@k.transpose(2,3))*self.scale #(B,num_heads,N,N)
        atten=atten.softmax(dim=-1) #对注意力分数进行归一化
        atten=self.atten_drop(atten) #注意力权重的 dropout 层，用于防止过拟合
        atten=(atten@v) #(B,num_heads,N,C//num_heads)

        atten=atten.transpose(-2,-1).reshape(B,N,C) # (B,N,C)

        atten=self.proj_AfterCatResult(atten)
        atten=self.proj_drop(atten)

        return atten

