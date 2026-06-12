"""
models/proposed_cnn_model.py
----------------------------
Proposed CNN: A CNN architecture faithfully reproduced from the model
proposed by Acici et al. in "Femoral neck fracture detection in X-ray
images using deep learning and genetic algorithm approaches" (2020).

Architecture:
    - 5 convolutional blocks (each: Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D)
    - Flatten after the final block
    - Dropout (p=0.5)
    - Fully connected classification layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProposedCNN(nn.Module):
    """
    CNN architecture based on the model proposed by Acici et al. (2020)
    for fracture detection with genetically optimized hyperparameters.

    Architecture:
        - 5 blocks (each: Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D)
        - Flatten after the final block
        - Dropout (p=0.5)
        - Fully connected classification layer
    """
    def __init__(
        self,
        num_classes: int = 10,
        base_filters: int = 8,  # Filter counts (e.g., 8,16,32,32,32) are optimized as in the reference.
        kernel_size: int = 3,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2
        out_ch2 = base_filters * 2
        self.block2 = nn.Sequential(
            nn.Conv2d(base_filters, out_ch2, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3
        out_ch3 = out_ch2 * 2
        self.block3 = nn.Sequential(
            nn.Conv2d(out_ch2, out_ch3, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4
        out_ch4 = out_ch3
        self.block4 = nn.Sequential(
            nn.Conv2d(out_ch3, out_ch4, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 5
        out_ch5 = out_ch4
        self.block5 = nn.Sequential(
            nn.Conv2d(out_ch4, out_ch5, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Adaptive pooling for input-size independence before FC layers
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_ch5, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x)
        return x


def build_proposed_cnn(
    num_classes: int = 10,
    base_filters: int = 8,
    kernel_size: int = 3,
    dropout_rate: float = 0.5,
):
    return ProposedCNN(
        num_classes=num_classes,
        base_filters=base_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    )


def get_optimizer(model, optimizer_name: str, learning_rate: float):
    """
    Returns AdamW with weight_decay=1e-3. The optimizer_name argument is
    accepted for interface compatibility but is not used by this model.
    """
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
