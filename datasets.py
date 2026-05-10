"""
Dataset Loaders
Supports CIFAR-10, CelebA, custom image folders, and tabular CSV datasets.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CelebA, ImageFolder
from PIL import Image
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Image Transforms
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int = 32, augment: bool = True):
    """Build training augmentation pipeline."""
    transforms = [T.Resize((image_size, image_size))]
    if augment:
        transforms += [
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    transforms += [
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ]
    return T.Compose(transforms)


def get_val_transforms(image_size: int = 32):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ─────────────────────────────────────────────
#  Built-in Datasets
# ─────────────────────────────────────────────

def get_cifar10(
    root: str = "data/cifar10",
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-10 train/val dataloaders."""
    train_ds = CIFAR10(root=root, train=True, download=True,
                        transform=get_train_transforms(image_size))
    val_ds = CIFAR10(root=root, train=False, download=True,
                      transform=get_val_transforms(image_size))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


def get_celeba(
    root: str = "data/celeba",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """CelebA train/val dataloaders."""
    train_ds = CelebA(root=root, split="train", download=True,
                       transform=get_train_transforms(image_size))
    val_ds = CelebA(root=root, split="valid", download=True,
                     transform=get_val_transforms(image_size))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


# ─────────────────────────────────────────────
#  Custom Image Folder Dataset
# ─────────────────────────────────────────────

class CustomImageDataset(Dataset):
    """
    Load images from a directory structure:
        data/
          class_a/img1.png
          class_b/img2.png
    or flat (no subdirs for unconditional training).
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, root: str, image_size: int = 64, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform or get_train_transforms(image_size)

        # Collect all image paths
        self.paths = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ])
        assert len(self.paths) > 0, f"No images found in {root}"

        # Build class labels from subdirs
        subdirs = sorted(set(p.parent.name for p in self.paths))
        self.class_to_idx = {c: i for i, c in enumerate(subdirs)}
        self.labels = [self.class_to_idx[p.parent.name] for p in self.paths]
        self.num_classes = len(subdirs)

        print(f"  Dataset: {len(self.paths)} images | {self.num_classes} classes")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────
#  Tabular Dataset
# ─────────────────────────────────────────────

class TabularDataset(Dataset):
    """
    Wraps a pandas DataFrame as a PyTorch Dataset.
    Normalizes features to zero mean / unit std.
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None, normalize: bool = True):
        self.target_col = target_col
        self.feature_cols = [c for c in df.columns if c != target_col]

        features = df[self.feature_cols].values.astype(np.float32)
        if normalize:
            self.mean = features.mean(0)
            self.std = features.std(0) + 1e-8
            features = (features - self.mean) / self.std

        self.features = torch.tensor(features)
        self.labels = None
        if target_col and target_col in df.columns:
            self.labels = torch.tensor(df[target_col].values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

    @classmethod
    def from_csv(cls, path: str, target_col: Optional[str] = None) -> "TabularDataset":
        df = pd.read_csv(path)
        return cls(df, target_col)


# ─────────────────────────────────────────────
#  Dataloader factory
# ─────────────────────────────────────────────

def build_dataloaders(config: dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train + val dataloaders from config dict.

    Config keys:
        dataset:    "cifar10" | "celeba" | "custom"
        data_root:  path to data directory
        image_size: int
        batch_size: int
        num_workers: int
    """
    ds_name = config.get("dataset", "cifar10")
    bs = config.get("batch_size", 128)
    nw = config.get("num_workers", 4)
    sz = config.get("image_size", 32)
    root = config.get("data_root", f"data/{ds_name}")

    if ds_name == "cifar10":
        return get_cifar10(root, sz, bs, nw)
    elif ds_name == "celeba":
        return get_celeba(root, sz, bs, nw)
    elif ds_name == "custom":
        ds = CustomImageDataset(root, sz)
        n_val = max(1, int(len(ds) * 0.1))
        n_train = len(ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
        return train_dl, val_dl
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
