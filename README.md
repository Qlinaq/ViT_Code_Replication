# ViT_Code_Replication
简单复现ViT with ViT—Base-B/16. 

This is a project for replicating the Vision Transformer (ViT) base model (ViT-B/16). The project includes Chinese and English comments for ease of study and reference。

ViT_smaller.py (Lightweight Vision Transformer)

This implementation provides a compact Vision Transformer (ViT-S/16-like) for efficient image classification on resource-constrained devices (e.g., Apple Silicon, laptops, or Colab Free GPU).

Model Structure:
Patch size: 16×16
Embedding dimension: 384
Number of transformer layers (depth): 8
Number of attention heads: 6
MLP hidden ratio: 4
Classification head: standard linear layer
Parameter Count: ~21 million (for CIFAR-10/100, using 384-dim, 8 layers)
Input Resolution: Configurable (default: 128×128, suitable for small datasets)
Training Resource: Runs efficiently on Mac M1/M2 (MPS backend) or Google Colab T4/V100 GPU; batch size 16–32 for best compatibility.
Typical training script:
train_cifar_vit_small.py supports both CIFAR-10 and CIFAR-100, with label smoothing and cosine LR scheduler.

Intended for:

Quick prototyping and research on small-to-medium datasets
Experiments in low-memory environments
Education and reproducible transformer research


