"""
utils/data_loader.py  —  DAGM 2007
------------------------------------
DAGM görüntüleri: grayscale 8-bit PNG, 512×512.
Pretrained modeller RGB bekliyor → Grayscale(3) ile 3 kanala çeviriyoruz.
10 sınıf (Class1 ... Class10).
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

CLASSES     = [f"Class{i}" for i in range(1, 11)]
NUM_CLASSES = 10

BASE_DIR  = Path("data/DAGM2007")
TRAIN_DIR = str(BASE_DIR / "train")
VAL_DIR   = str(BASE_DIR / "val")
TEST_DIR  = str(BASE_DIR / "test")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_cache = {}


def get_loaders(img_size: int = 224, batch_size: int = 32, num_workers: int = 0):
    """
    DAGM 2007 DataLoader döndürür.
    Grayscale PNG → 3 kanallı tensor (ImageNet normalize).
    """
    key = (img_size, batch_size)
    if key in _cache:
        return _cache[key]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),  # grayscale → RGB
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    class_names = list(train_ds.class_to_idx.keys())
    result      = (train_loader, val_loader, test_loader, class_names)
    _cache[key] = result
    return result
