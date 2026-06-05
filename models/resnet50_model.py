"""
models/resnet50_model.py
------------------------
Model 2: ResNet50 Transfer Learning (PyTorch / torchvision)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(
    num_classes:  int   = 6,
    dropout_rate: float = 0.4,
    dense_units:  int   = 512,
    unfreeze_blocks: int = 2,   # Son kaç ResNet bloğu unfreeze edilsin (1-4)
):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Tüm parametreleri dondur
    for param in model.parameters():
        param.requires_grad = False

    # Son N layer4 bloğunu unfreeze et
    blocks = list(model.layer4.children())
    for block in blocks[-unfreeze_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    # FC başlığını değiştir
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, dense_units),
        nn.BatchNorm1d(dense_units),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, dense_units // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate * 0.5),
        nn.Linear(dense_units // 2, num_classes),
    )

    return model


def get_optimizer(model, optimizer_name: str, learning_rate: float):
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9,
                               nesterov=True, weight_decay=1e-4)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=learning_rate, weight_decay=1e-4)
    return torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
