"""
dataset.py

PyTorch Dataset for CICIDS2017 processed features.
Supports train/val/test splits with stratification.

Usage:
    from src.dataset import CICIDSDataset, get_dataloaders
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

PROCESSED_DIR = "data/processed"


class CICIDSDataset(Dataset):
    """Flow-level feature dataset for CICIDS2017."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def num_features(self) -> int:
        return self.X.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1


def load_processed(processed_dir: str = PROCESSED_DIR):
    """Load preprocessed numpy arrays from disk."""
    X = np.load(os.path.join(processed_dir, "X.npy"))
    y = np.load(os.path.join(processed_dir, "y.npy"))
    return X, y


def get_dataloaders(
    batch_size: int = 512,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 4,
    processed_dir: str = PROCESSED_DIR,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Returns (train_loader, val_loader, test_loader, num_features, num_classes).

    Stratified split so all attack classes appear in every split.
    """
    X, y = load_processed(processed_dir)

    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Then split train/val from the remainder
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_frac,
        stratify=y_train_val,
        random_state=random_state,
    )

    print(f"Split sizes — train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")

    train_ds = CICIDSDataset(X_train, y_train)
    val_ds = CICIDSDataset(X_val, y_val)
    test_ds = CICIDSDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds.num_features, train_ds.num_classes
