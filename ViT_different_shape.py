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
    def __init__(self,input_shape=[224,224],patch_size=16,in_channels=3,num_features=768, norm_layer=None,flattn=True):
        super(PatchEmbed,self).__init__()
        self.num_patches=(input_shape[0]//patch_size) * (input_shape[1]//patch_size) #not / but //
        self.proj=nn.Conv2d(in_channels,num_features,kernel_size=patch_size,stride=patch_size,)#定义卷积层，
        #将输入图像 （224x224）分割成 patch_size x patch_size 的小块，同时proj到 一个高维空间num_features 维度
        #卷积后：[B, num_features, H/patch_size, W/patch_size]
        self.flattn=flattn #展平后：[B, num_features, num_patches]
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
        if self.flattn:
            # 将特征图展平并将维度转换
            # 形状变化: (batch, 768, 14, 14) -> (batch, 768, 196) -> (batch, 196, 768)
            x=x.flattn(2).transpose(1,2)
        # 对嵌入后的块进行归一化处理
        x=self.norm(x)

        return x
'''
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
        self.dropout=nn.Dropout(p=drop_rate)         # 应用 dropout 正则化，请注意这里的 `droupout` 可能是 `dropout` 的拼写错误

    def forward(self,x):
        """
        前向传播函数
        :param x: 输入张量，形状为 (batch, num_patches+1, embedding_dim)
        :return: 添加位置编码后的输出张量，形状为 (batch, num_patches+1, embedding_dim)
        """
        x=x+self.pos_embed #将位置编码加到输入张量上
        x=self.dropout(x)

        return x '''
        

#class EncoderBlock(nn.Module):

class attntion(nn.Module):
    def __init__(self,num_heads=8,dim_of_patch=768,qkv_bias=False,attn_drop_rate=0.1,proj_drop_rate=0.1):
        super(attntion,self).__init__()
        self.num_heads=num_heads
        self.scale=(dim_of_patch//num_heads)**(-0.5) #缩放因子，用于缩放注意力分数，防止 softmax 函数输出过大或过小的数值
        self.qkv=nn.Linear(dim_of_patch,3*dim_of_patch,bias=qkv_bias) #线性变换层，用于生成查询（Q）、键（K）、值（V）向量
        self.attn_drop=nn.Dropout(attn_drop_rate) #注意力权重的 dropout 层，用于防止过拟合
        self.proj=nn.Linear(dim_of_patch,dim_of_patch)
        self.proj_drop=nn.Dropout(proj_drop_rate)

    def forward(self,x):
        # 获取输入张量的形状: (B: Batch size, N: Sequence length, C: Embedding dimension)
        B,N,C=x.shape
        # 1. 通过一个大的线性层一次性生成所有heads的 Q, K, V
        # 输入 x 形状: (B, N, C)
        # 输出 attn_x 形状: (B, N, 3 * C)
        attn_x=self.qkv(x)
        # 2. 重塑(reshape)并变维(permute)以分离 Q, K, V 并为多头准备
        # 目标形状: (3, B, num_heads, N, head_dim) where head_dim = C // num_heads
        attn_x=attn_x.reshape(B,N,3,self.num_heads,C//self.num_heads)
        attn_x=attn_x.permute(2,0,3,1,4)
        # 3. 分别获取 Q, K, V
        # q, k, v 形状: (B, num_heads, N, head_dim)
        q,k,v=attn_x[0],attn_x[1],attn_x[2]


        # 4. 计算注意力分数 (Scaled Dot-Product attntion)
        # (q @ k.transpose(-2, -1)) -> (B, num_heads, N, N)
        attn=(q@k.transpose(-2,-1))*self.scale
        # 5. 对注意力分数在最后一个维度上进行 softmax 归一化
        attn=attn.softmax(dim=-1)
        # 6. 对注意力分数应用 dropout 正则化，防止过拟合
        attn=self.attn_drop(attn)
        # 7. 将注意力权重与 V 相乘，得到加权的 V
        # (attn @ v) -> (B, num_heads, N, head_dim)
        attn=(attn@v)

        # 8. 重塑(reshape)输出，将多头结果拼接起来
        # transpose(1, 2) -> (B, N, num_heads, head_dim)
        # reshape(B, N, C) -> (B, N, C)
        attn=attn.transpose(1,2).reshape(B,N,C)

        # 9. 通过最终的线性层和 dropout 层
        attn=self.proj(attn)
        attn=self.proj_drop(attn)

        return attn

def GELU_fn(x):
    # result=0.5 * x * (1 + torch.erf(x / math.sqrt(2))) #实际的GELU激活函数
    result=0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3)))) #近似GELU激活函数方便计算
    return result



class mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=GELU_fn,drop_probs=(0., 0.)):
        """
        初始化 mlp(多层感知机)模块。

        Args:
            in_features (int): 输入特征的维度。
            hidden_features (int, optional): 隐藏层特征的维度。如果为 None，则默认为 `in_features`。
            out_features (int, optional): 输出特征的维度。如果为 None，则默认为 `in_features`。
            act_layer (nn.Module, optional): 使用的激活函数。默认为 `nn.GELU`。
            drop_probs (tuple[float], optional): 两个 Dropout 层的丢弃率。默认为 `(0., 0.)`。
        """
        super(mlp,self).__init__()
        # 如果未指定输出特征维度，则默认等于输入特征维度
        out_features=out_features or in_features
        # 如果未指定隐藏层特征维度，则默认等于输入特征维度
        hidden_features=hidden_features or in_features
        
        # 第一个全连接层
        self.fc1=nn.Linear(in_features,hidden_features)
        # 激活函数
        self.act=act_layer
        # 第二个全连接层
        self.fc2=nn.Linear(hidden_features,out_features)

        # 定义两个 Dropout 层，可以为它们设置不同的丢弃率
        self.drop1=nn.Dropout(drop_probs[0])
        self.drop2=nn.Dropout(drop_probs[1])

    def forward(self,x):
        #x=batch, 197, 768

        x=self.fc1(x) #batch, 197, 768-> batch, 197, 3072
        
        x=self.act(x)#batch, 197, 3072 -> batch, 197, 3072
        
        x=self.drop1(x)#batch, 197, 3072 -> batch, 197, 3072
        
        x=self.fc2(x)#batch, 197, 3072 -> batch, 197, 768
        
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
    shape_for_rand=(x.shape[0],)+(x.ndim-1)*(1,)

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
        super(Drop_Path,self).__init__()
        self.drop_prob=drop_prob

    def forward(self,x):
        return drop_path(x,self.drop_prob,self.training)

class EncoderBlock(nn.Module):
    def __init__(self,in_features,num_heads,mlp_ratio=4,qkv_bias=False,drop_probs=(0.,0.),
                 attn_drop_rate=0.,proj_drop_rate=0.,drop_path_rate=0.,norm_layer=nn.LayerNorm):
        super(EncoderBlock,self).__init__()
        self.attn=attntion(num_heads=num_heads,dim_of_patch=in_features,qkv_bias=qkv_bias,attn_drop_rate=attn_drop_rate,proj_drop_rate=proj_drop_rate)
        self.norm1=norm_layer(in_features)
        self.mlp=mlp(in_features=in_features,hidden_features=int(in_features*mlp_ratio),out_features=in_features,act_layer=GELU_fn,drop_probs=drop_probs)
        self.norm2=norm_layer(in_features)
        self.drop_path=Drop_Path(drop_path_rate) if drop_path_rate>0. else nn.Identity()

    def forward(self,x):
        #residual connection
        x=self.drop_path(self.attn(self.norm1(x)))+x
        x=self.drop_path(self.mlp(self.norm2(x)))+x
        return x

class ViT(nn.Module):
    # ViT模型的实现
    def __init__(self,input_shape=[224,224],patch_size=16,in_channels=3,num_classes=1000,num_features=768,
                 depth=12,drop_rate=0.1,num_heads=12,mlp_ratio=4., qkv_bias=True,attn_drop_rate=0.1,proj_drop_rate=0.1,drop_path_rate=0.1,
                 act_layer=GELU_fn,norm_layer=partial(nn.LayerNorm,eps=1e-6)):
        super(ViT,self).__init__()
        
        self.new_feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]
        self.old_feature_shape=[int (224//patch_size),int(224//patch_size)]
        num_patches_old = self.old_feature_shape[0] * self.old_feature_shape[1]
#1.Patch嵌入
        self.patch_embed=PatchEmbed(input_shape=input_shape,patch_size=patch_size,in_channels=in_channels,num_features=num_features,norm_layer=None,flattn=True)     
        self.num_patches=self.patch_embed.num_patches
#2.添加CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features))
#3.添加位置编码self.pos_embed=Add_Position_Embed(num_patches=self.num_patches,embed_dim=num_features,drop_rate=drop_rate)
        #位置编码是一个可学习的参数，形状为 (1, num_patches + 1, num_features)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_old + 1, num_features))
        self.pos_drop = nn.Dropout(p=drop_rate)

#4.为每层depth的EncoderBlock添加drop_path_rate
#linspace 函数用于生成一系列等差数列，这里用于生成 depth 个 drop_path_rate 值，并将它们堆叠成一个列表。
        drop_path_rates=torch.linspace(0,drop_path_rate,depth).tolist()

#5.EncoderBlock的堆叠
        block=[]
        for i in range(depth):
            block.append(
                EncoderBlock(in_features=num_features,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                             drop_probs=(drop_rate,drop_rate),attn_drop_rate=attn_drop_rate,proj_drop_rate=proj_drop_rate,
                             drop_path_rate=drop_path_rates[i],norm_layer=norm_layer)
            )
        self.blocks=nn.ModuleList(block)

#6.Final normalization layer
        self.norm=norm_layer(num_features)
#7.Classification head
        # 分类头，将特征表征从高维（如 768）映射到分类任务所需的输出维度（100 个类别）
        # [batch_size, num_features]-> [batch_size, num_classes]
        self.head=nn.Linear(num_features,num_classes)  if num_classes>0 else nn.Identity()
#8.简单初始化参数
        self.apply(self._init_weights)

    def _init_weights(self,m):
        """
        初始化模型中不同模块的权重。
        这个方法会被 `model.apply(_init_weights)` 调用，递归地应用到模型的所有子模块上。

        Args:
            m (nn.Module): 需要初始化权重的模块。
        """
        # isinstance() 函数用于检查一个对象是否是一个已知的类型。
        if isinstance(m,nn.Linear):
            # 如果是线性层 (nn.Linear)，则使用截断正态分布来初始化权重。
            # 均值为 0，标准差为 0.02。
            # 这是 ViT 论文中推荐的初始化方法。
            nn.init.trunc_normal_(m.weight,std=0.02)
            # 如果线性层有偏置项 (bias)，则将其初始化为 0。
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.LayerNorm):
            # 如果是层归一化 (LayerNorm)，则将其权重初始化为 1，偏置初始化为 0。
            # 这使得 LayerNorm 在初始时近似于一个恒等变换。
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Conv2d):
            # 如果是二维卷积层 (Conv2d)，则使用 Kaiming 正态分布初始化权重。
            # 'fan_out' 模式保留了反向传播时权重的方差。
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            # 如果卷积层有偏置项，则将其初始化为 0。
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if isinstance(self.cls_token, nn.Parameter):
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self,x):
        #1.Patch嵌入
        x=self.patch_embed(x) # [B, 3, 224, 224] -> [B, 196, 768]
        #2.添加CLS Token

        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        #[B, 1, 768] + [B, 196, 768] -> [B, 197, 768]
        x = torch.cat((cls_token, x), dim=1)  
        
        # 3.位置编码插值 self.pos_embed=[1, 197, 768]
        # 3.1 拆分cls_token和patch_token
        cls_token_pe=self.pos_embed[:,0:1,:] #[1, 197, 768]-> [1, 1，768]
        image_token_pe=self.pos_embed[:,1:,:] #[1, 197, 768]-> [1, 196，768]
        N_old = image_token_pe.shape[1]
        
        # 3.2 把图像位置编码还原为网格，准备插值
        # 1, 196, 768 -> 1, 14, 14, 768 -> 1, 768, 14, 14
        image_token_pe=image_token_pe.view(1,*self.old_feature_shape,-1).permute(0,3,1,2)
        
        # 3.4 插值 到新的图像尺寸 以防止输入图像不是224x224
        #F.interpolate 函数用于对输入张量进行插值操作，这里用于将图像位置编码从 14x14 插值到新的尺寸（如 16x16）。
        # 这是为了处理输入图像尺寸不是 224x224 的情况，确保位置编码与输入图像的特征图尺寸匹配。
        #1, 768, 14, 14 -> 1, 768, new_H, new_W
        image_token_pe=F.interpolate(image_token_pe,size=self.new_feature_shape,mode='bicubic',align_corners=False)
        
        # 3.5 恢复形状
        # 1, 768, new_H, new_W -> 1, 196, 768
        image_token_pe=image_token_pe.permute(0,2,3,1).flattn(1,2)

        # 3.6 拼接cls_token和patch_token
        # [1, 1, 768] + [1, 196, 768] -> [1, 197, 768]
        pos_embed=torch.cat((cls_token_pe,image_token_pe),dim=1) 

        # 3.7 添加位置编码并进行 dropout
        x=x+pos_embed # [B, 197, 768] + [1, 197, 768] -> [B, 197, 768] 用到了broadcast机制
        x=self.pos_drop(x) # [B, 197, 768]


        #4.通过多个EncoderBlock
        #nn.ModuleList 需要手动遍历,更灵活可以添加一些操作
        #nn.Sequential 可以自动遍历
        for block in self.blocks: 
            x=block(x) # [B, 197, 768] -> [B, 197, 768]
        #5.最终的归一化层
        x=self.norm(x) # [B, 197, 768] -> [B, 197, 768]
        #6.分类头   
        cls=x[:,0] # 只取CLS Token对应的特征表示 [B, 197, 768] -> [B, 768]
        return cls
    
    def forward(self,x):
        x=self.forward_features(x) # [B, 3, 224, 224] -> [B, 768]
        x=self.head(x)            # [B, 768] -> [B, 1000]
        return x
    
    #def freeze_backbone(x):
        #冻结 backbone（例如前 8 层）只训练头部或少量层，稳定训练、节省算力
    



def vit(input_shape=[224,224],pretrained=False,num_classes=1000):
    """
    构建并返回一个 Vision Transformer (ViT) 模型。

    这个函数是一个工厂函数，可以方便地创建一个 ViT 模型，并选择性地加载预训练权重以及调整分类头的类别数。

    Args:
        input_shape (list, optional): 输入图像的尺寸 [height, width]。默认为 [224, 224]。
        pretrained (bool, optional): 如果为 True,则加载预训练的权重。默认为 False。
        num_classes (int, optional): 输出分类的类别数。默认为 1000 (ImageNet 的类别数)。

    Returns:
        nn.Module: 配置好的 ViT 模型实例。
    """
    # 1. 创建一个 ViT 模型实例，并将其移动到指定的计算设备 (device) 上。
    #    注意: `device` 应该是在此函数外部定义的全局变量 (例如, "cuda" 或 "cpu")。
    model=ViT(input_shape=input_shape).to(device)

    # 2. 如果 `pretrained` 参数为 True，则从本地文件 "vit_base_patch16_224.pth" 加载预训练权重。
    if pretrained:
        model.load_state_dict(torch.load("vit-patch_16.pth"))
        

    # 3. 如果指定的 `num_classes` 不是默认的 1000，则替换模型的分类头 (model.head)。
    #    创建一个新的线性层，其输出维度为 `num_classes`,并将其移动到 `device`。
    if num_classes!=1000:
        model.head=nn.Linear(768,num_classes).to(device)

    # 4. 返回最终配置好的模型。
    return model

if __name__=='__main__':
    model=vit(input_shape=[224,224],pretrained=True,num_classes=1000)
    print(model)