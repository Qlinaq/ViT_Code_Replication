import torch
from ViT_Base16 import PatchEmbed, Add_CLS_Token

def verify_patch_embed():
    """
    验证PatchEmbed类的前向传播函数是否正确
    
    Returns:
        tuple: (output_tensor, output_shape) 输出张量和其形状
    """
    # 创建PatchEmbed实例
    patch_embed = PatchEmbed(input_shape=[224, 224], patch_size=16, in_channels=3, num_features=768)
    
    # 创建随机输入张量
    x = torch.randn(1, 3, 224, 224)
    
    # 进行前向传播
    output = patch_embed(x)
    
    # 打印输出形状
    print("Output shape:", output.shape)
    
    return output, output.shape

def verify_add_cls_token():
    """
    验证Add_CLS_Token类的前向传播函数是否正确
    
    Returns:
        tuple: (output_tensor, output_shape) 输出张量和其形状
    """
    # 创建Add_CLS_Token实例
    add_cls_token = Add_CLS_Token(embed_dim=768)
    
    # 创建随机输入张量
    x = torch.randn(1, 196, 768)
    
    # 进行前向传播
    output = add_cls_token(x)
    
    # 打印输出形状
    print("Output shape:", output.shape)
    
    return output, output.shape

if __name__ == "__main__":
    print("验证PatchEmbed:")
    output_patch, shape_patch = verify_patch_embed()
    print("\n验证Add_CLS_Token:")
    output_cls, shape_cls = verify_add_cls_token()
