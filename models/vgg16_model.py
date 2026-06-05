"""
models/vgg16_model.py
---------------------
Model 1: VGG16 Transfer Learning (PyTorch / torchvision)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_vgg16(
    num_classes:    int   = 6,
    dropout_rate:   float = 0.5,
    dense_units:    int   = 256,
    fine_tune_layers: int = 4,
    freeze_features: bool = True,
):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Feature katmanlarını dondur
    for param in model.features.parameters():
        param.requires_grad = False

    # Son N konv bloğunu unfreeze et
    children = list(model.features.children())
    for layer in children[-fine_tune_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Classifier başlığını değiştir
    in_features = model.classifier[0].in_features  # 25088
    model.classifier = nn.Sequential(
        nn.Linear(in_features, dense_units),
        nn.BatchNorm1d(dense_units),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, num_classes),
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
