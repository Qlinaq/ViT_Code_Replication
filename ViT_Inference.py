import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import json

# ====== 1. 加载 ViT 模型 ======
# 导入模型构造函数
from ViT_different_shape import vit, device

# 设置输入尺寸
input_shape = [231, 240]

# 加载模型并加载预训练权重
model = vit(input_shape=input_shape, pretrained=True, num_classes=1000).to(device)
model.eval()

# ====== 2. 加载并预处理图片 ======
img_path = "picture_for_understanding/Main_Struture.jpg"   

# ViT 支持任意尺寸，按实际尺寸resize即可
transform = transforms.Compose([
    transforms.Resize(input_shape),     # 保持和模型输入一致
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 231, 240]

# ====== 3. 推理 ======
with torch.no_grad():
    output = model(input_tensor)        # shape: [1, 1000]

# ====== 4. 解析结果 ======
probabilities = F.softmax(output, dim=1)
predicted_class_idx = probabilities.argmax(dim=1).item()
predicted_prob = probabilities[0, predicted_class_idx].item()

# ====== 5. 映射类别索引到类名（ImageNet） ======
# 下载并放置 imagenet_class_index.json 文件到本目录
# 链接: https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet_class_index.json

with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    class_name = class_idx[str(predicted_class_idx)][1]

print(f"Predicted class index: {predicted_class_idx}")
print(f"Predicted class name: {class_name}")
print(f"Predicted probability: {predicted_prob:.4f}")