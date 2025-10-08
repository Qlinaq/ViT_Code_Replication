import torch
from ViT_smaller_SPT_LSA import vit_small

# 配置
input_shape = [256 , 256]   # 输入图片分辨率
num_classes = 10           
device = torch.device("mps" if torch.backends.mps.is_available() else (
                     "cuda:0" if torch.cuda.is_available() else "cpu"))

model = vit_small(input_shape=input_shape, num_classes=num_classes)
model.eval()  
model.to(device)

# 构造一张随机图片（标准RGB范围，形状[B, C, H, W]）
B = 1
img = torch.rand(B, 3, input_shape[0], input_shape[1], device=device)

# 前向推理
with torch.no_grad():
    logits = model(img)  # [B, num_classes]
    pred = logits.argmax(dim=-1)  # [B]

print("logits:", logits)
print("predicted class:", pred.item())