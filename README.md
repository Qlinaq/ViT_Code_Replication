# ViT_Code_Replication
简单复现
ViT with ViT—Base-B/16. 

This is a project for replicating the Vision Transformer (ViT) base model (ViT-B/16). The project includes Chinese and English comments for ease of study and reference。

# ViT_Smaller and ViT-B/16: Vision Transformers for Image Classification

This project implements two Vision Transformer (ViT) models for image classification:  
- **ViT_Smaller**: a lightweight ViT-S/16 style model for efficient experiments on resource-constrained devices.  
- **ViT-B/16**: the canonical base ViT model from [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), implemented in [`ViT_different_shape.py`](./ViT_different_shape.py).

Both models support image classification on datasets like CIFAR-10/100 and can run on Apple Silicon (MPS), Google Colab GPUs, or standard CUDA/CPU environments.

---

## 📦 Model Architectures

### ViT_Smaller

- **Patch size:** 16×16  
- **Embedding dim:** 384  
- **Transformer layers (Depth):** 8  
- **Attention heads:** 6 (64 dim per head)  
- **MLP ratio:** 4  
- **Parameter count:** ~21M (CIFAR-10/100)  
- **Input resolution:** Configurable (default 128×128, with position embedding interpolation)

### ViT-B/16 (`ViT_different_shape.py`)

- **Patch size:** 16×16  
- **Embedding dim:** 768  
- **Transformer layers (Depth):** 12  
- **Attention heads:** 12  
- **MLP ratio:** 4  
- **Parameter count:** ~86M  
- **Input resolution:** Flexible (default 224×224, position embedding interpolates automatically)  
- **Implements:** The architecture from [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## 🚀 Training Setup for ViT_Smaller

- **Dataset:** CIFAR-10 (can switch to CIFAR-100)  
- **Input size:** 128×128 (configurable)  
- **Batch size:** 32 (adjustable for hardware)  
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.05)  
- **Learning rate schedule:** CosineAnnealingLR  
- **Label smoothing:** 0.1  
- **Epochs:** 20  
- **Environment:** Colab T4, Mac M2 Pro (MPS), CUDA, CPU

---

## 📊 Training Results (Colab GPU)

Below is a screenshot of 20 epochs training on Colab + T4 GPU:

![ViT Smaller Training Log](attachment:image)

- **Final validation accuracy (Val Acc):** `0.6451` (64.51%)
- **Loss:** Both train and validation loss decrease steadily as accuracy increases, indicating effective learning.

---

## ⚠️ Note on Mac M2 (MPS) & ViT-B/16

> **Warning:**  
> The base ViT-B/16 model (`ViT_different_shape.py`) is large and **may cause a `RuntimeError: MPS backend out of memory`** on Mac M2 or other Apple Silicon devices, especially at higher resolutions or batch sizes.  
>  
> For reliable training/inference on Mac M2, use **ViT_Smaller** or reduce batch size and input resolution for the base model.

---

## 🛠️ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-repo-url/ViT_Smaller.git
cd ViT_Smaller

# 2. Install dependencies
pip install torch torchvision

# 3. Train ViT_Smaller
python train_cifar_vit_small.py
```

- **Change dataset:** Edit `dataset_name` in `train_cifar_vit_small.py` to `"CIFAR100"`
- **Change input size:** Edit `input_shape=(128,128)`

---

## 🗂️ Code Structure

| File                        | Description                                                        |
|-----------------------------|--------------------------------------------------------------------|
| `ViT_small.py`              | Implementation of ViT_Smaller (384 dim, 8 layers, 6 heads)         |
| `train_cifar_vit_small.py`  | Training script for CIFAR-10/100 (includes augmentation, scheduler)|
| `ViT_different_shape.py`    | Full ViT-B/16 implementation (12 layers, 768 dim, flexible input)  |

---

## 🔎 Inference with ViT-B/16

You can use `ViT_different_shape.py` to run inference on custom images:

```python
from ViT_different_shape import vit
import torch

# Example: Single image tensor with shape [1, 3, 224, 224]
model = vit(input_shape=[224, 224], pretrained=True, num_classes=1000)
model.eval()

with torch.no_grad():
    logits = model(your_image_tensor.to(next(model.parameters()).device))
    predicted_class = logits.argmax(dim=-1)
```
- **Input shape is flexible:** The model automatically resizes position embeddings for different image sizes.

---

## 🔁 Reproducibility

> Training results may show small variations across devices (MPS/Mac vs. CUDA/GPU), which is expected.

To maximize reproducibility, set a fixed random seed:

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 📒 References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/)

---

Feel free to request an extended English version, experiment report, or additional result tables!
```
