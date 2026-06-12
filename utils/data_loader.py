"""
utils/data_loader.py
--------------------
Data loading utilities for the DAGM 2007 texture dataset.

DAGM images are grayscale 8-bit PNG files at 512x512 resolution.
Since pretrained models expect RGB (3-channel) input, grayscale images
are replicated to 3 channels via Grayscale(num_output_channels=3).
The dataset contains 10 classes (Class1 through Class10).
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import random
import numpy as np

CLASSES     = [f"Class{i}" for i in range(1, 11)]
NUM_CLASSES = 10

BASE_DIR  = Path("data/DAGM2007")
TRAIN_DIR = str(BASE_DIR / "train")
VAL_DIR   = str(BASE_DIR / "val")
TEST_DIR  = str(BASE_DIR / "test")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_cache = {}

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def get_loaders(img_size: int = 224, batch_size: int = 32, num_workers: int = 0):
    """
    Construct PyTorch DataLoaders for the DAGM 2007 dataset.

    Grayscale PNG images are converted to 3-channel tensors and normalized
    using ImageNet statistics. A stratified subset of the training set is
    used (limited samples per class) to increase classification difficulty
    and better evaluate optimization methods.

    Parameters
    ----------
    img_size : int
        Target spatial resolution for resizing (default: 224).
    batch_size : int
        Mini-batch size for all loaders (default: 32).
    num_workers : int
        Number of data loading worker processes (default: 0).

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, class_names)
    """
    key = (img_size, batch_size)
    if key in _cache:
        return _cache[key]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=0.05),  # Gaussian noise for regularization
        transforms.RandomErasing(p=0.5),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    full_train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=val_tf)

    # Stratified subsampling: select a limited number of samples per class
    # to prevent trivially high accuracy and better differentiate optimizers.
    samples_per_class = 50
    class_indices = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, target) in enumerate(full_train_ds.samples):
        class_indices[target].append(idx)
        
    subset_indices = []
    for c in range(NUM_CLASSES):
        # Use all available samples if fewer than the limit; otherwise sample randomly
        available = len(class_indices[c])
        selected_count = min(samples_per_class, available)
        subset_indices.extend(random.sample(class_indices[c], selected_count))
        
    train_ds = Subset(full_train_ds, subset_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    class_names = list(full_train_ds.class_to_idx.keys())
    result      = (train_loader, val_loader, test_loader, class_names)
    _cache[key] = result
    return result
