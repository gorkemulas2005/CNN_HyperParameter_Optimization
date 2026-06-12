"""
models/custom_cnn_model.py
--------------------------
Model 3: Custom CNN built from scratch (PyTorch).

Architecture:
    - Depthwise Separable Convolution blocks
    - Residual shortcuts around separable convolution blocks
    - LeakyReLU activation with Batch Normalization
    - Global Average Pooling with Dropout

Note:
    SEBlock is defined in this module, but the current SepConvBlock forward
    path does not apply it. Existing experiment outputs therefore correspond
    to the residual depthwise-separable CNN path, not an active SE-attention
    variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Squeeze-and-Excitation Block ---------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        mid = max(1, channels // ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# -- Depthwise Separable Convolution Block ------------------------------------
class SepConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, use_se: bool = True):
        super().__init__()
        pad = kernel_size // 2

        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad,
                      groups=in_ch, bias=False),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.3),
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        # Residual: use 1x1 convolution when channel dimensions do not match
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# -- Custom CNN ----------------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(
        self,
        num_classes:  int   = 6,
        base_filters: int   = 32,
        num_blocks:   int   = 3,
        kernel_size:  int   = 3,
        dropout_rate: float = 0.5,
        use_se:       bool  = True,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_filters, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(2),
        )

        # Separable convolution blocks
        blocks = []
        in_ch  = base_filters
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            blocks.append(SepConvBlock(in_ch, out_ch, kernel_size, use_se))
            if i < num_blocks - 1:
                blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, in_ch),
            nn.BatchNorm1d(in_ch),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout_rate),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def build_custom_cnn(
    num_classes:  int   = 6,
    base_filters: int   = 32,
    num_blocks:   int   = 3,
    kernel_size:  int   = 3,
    dropout_rate: float = 0.5,
    use_se:       bool  = True,
):
    return CustomCNN(
        num_classes=num_classes,
        base_filters=base_filters,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        use_se=use_se,
    )


def get_optimizer(model, optimizer_name: str, learning_rate: float):
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate,
                               momentum=0.9, nesterov=True, weight_decay=1e-3)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
