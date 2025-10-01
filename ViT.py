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
        # 获取输入张量的形状: (B: Batch size, N: Sequence length, C: Embedding dimension)
        B,N,C=x.shape
        # 1. 通过一个大的线性层一次性生成所有heads的 Q, K, V
        # 输入 x 形状: (B, N, C)
        # 输出 Atten_x 形状: (B, N, 3 * C)
        Atten_x=self.big_qkv_with_W_for_heads(x)
        # 2. 重塑(reshape)并变维(permute)以分离 Q, K, V 并为多头准备
        # 目标形状: (3, B, num_heads, N, head_dim) where head_dim = C // num_heads
        Atten_x=Atten_x.reshape(B,N,3,self.num_heads,C//self.num_heads)
        Atten_x=Atten_x.permute(2,0,3,1,4)
        # 3. 分别获取 Q, K, V
        # q, k, v 形状: (B, num_heads, N, head_dim)
        q,k,v=Atten_x[0],Atten_x[1],Atten_x[2]


        # 4. 计算注意力分数 (Scaled Dot-Product Attention)
        # (q @ k.transpose(-2, -1)) -> (B, num_heads, N, N)
        atten=(q@k.transpose(-2,-1))*self.scale
        # 5. 对注意力分数在最后一个维度上进行 softmax 归一化
        atten=atten.softmax(dim=-1)
        # 6. 对注意力分数应用 dropout 正则化，防止过拟合
        atten=self.atten_drop(atten)
        # 7. 将注意力权重与 V 相乘，得到加权的 V
        # (atten @ v) -> (B, num_heads, N, head_dim)
        atten=(atten@v)

        # 8. 重塑(reshape)输出，将多头结果拼接起来
        # transpose(1, 2) -> (B, N, num_heads, head_dim)
        # reshape(B, N, C) -> (B, N, C)
        atten=atten.transpose(1,2).reshape(B,N,C)

        # 9. 通过最终的线性层和 dropout 层
        atten=self.proj_AfterCatResult(atten)
        atten=self.proj_drop(atten)

        return atten

def GELU(x):
    # result=0.5 * x * (1 + torch.erf(x / math.sqrt(2))) #实际的GELU激活函数
    result=0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3)))) #近似GELU激活函数方便计算
    return result



class MLP(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop_probs=(0.1, 0.1)):
        """
        初始化 MLP(多层感知机)模块。

        Args:
            in_features (int): 输入特征的维度。
            hidden_features (int, optional): 隐藏层特征的维度。如果为 None，则默认为 `in_features`。
            out_features (int, optional): 输出特征的维度。如果为 None，则默认为 `in_features`。
            act_layer (nn.Module, optional): 使用的激活函数。默认为 `nn.GELU`。
            drop_probs (tuple[float], optional): 两个 Dropout 层的丢弃率。默认为 `(0.1, 0.1)`。
        """
        super(MLP,self).__init__()
        # 如果未指定输出特征维度，则默认等于输入特征维度
        out_features=out_features or in_features
        # 如果未指定隐藏层特征维度，则默认等于输入特征维度
        hidden_features=hidden_features or in_features
        
        # 第一个全连接层
        self.l1=nn.Linear(in_features,hidden_features)
        # 激活函数
        self.act=act_layer()
        # 第二个全连接层
        self.l2=nn.Linear(hidden_features,out_features)

        # 定义两个 Dropout 层，可以为它们设置不同的丢弃率
        self.drop1=nn.Dropout(drop_probs[0])
        self.drop2=nn.Dropout(drop_probs[1])

    def forward(self,x):
        #x=batch, 197, 768

        x=self.l1(x) #batch, 197, 768-> batch, 197, 3072
        
        x=self.act(x)#batch, 197, 3072 -> batch, 197, 3072
        
        x=self.drop1(x)#batch, 197, 3072 -> batch, 197, 3072
        
        x=self.l2(x)#batch, 197, 3072 -> batch, 197, 768
        
        x=self.drop2(x)#batch, 197, 768 -> batch, 197, 768
        
        
        return x
    

def drop_path(x,drop_prob:float=0.,training:bool=False):
    """
    Drop Path (Stochastic Depth) implementation.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, ..., features).
        drop_prob (float): Probability of dropping the path.
        training (bool): Whether the model is in training mode.
        
    Returns:
        torch.Tensor: Output tensor after applying Drop Path.
    """
    #float=0. 是显示类型表示这是float类型的0 ，而不是默认的int类型的0
    # 当 drop_prob 为 0 或者模型不在训练模式时，直接返回输入张量，不进行 Drop Path 操作
    if drop_prob==0. or not training: 
        return x
    # 按照给定的丢弃率生成一个 0-1 之间的随机数
    keep_prob=1-drop_prob
    # 计算随机数的形状，以便在后续操作中进行广播
    shape_for_rand=x.shape[0]+(x.ndim-1)*(1,)

    # 生成与输入张量形状匹配的范围是[0,1)的随机张量,
    random_tensor=torch.rand(shape_for_rand,dtype=x.dtype,device=x.device)

    #加上keep_prob，方便后续的二值化
    random_tensor=random_tensor+keep_prob

    #二值化，生成0或1的张量,用来决定哪些样本被保留，以及后续对x的缩放
    random_tensor=random_tensor.floor() 

    #5 * 0.8 + 0 * 0.2 = 4，说明有80%的样本会被保留，20%的样本会被丢弃，0.8相当于keep_prob
    # 1.通过与随机张量相乘来二值化 x，相当于乘0.8
    #2.dropout后的x 通过除以keep_prob来缩放未被丢弃的样本，以保持期望值不变，相当于除0.8
    output=random_tensor* x.div(keep_prob)

    return output

class Drop_Path(nn.Module):
    def __init__(self,drop_prob=None):
        super(Drop_Path,self).__init()
        self.drop_prob=drop_prob

    def forward(self,x):
        return drop_path(x,self.drop_prob,self.training)