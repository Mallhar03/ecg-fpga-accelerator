import logging
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

logger = logging.getLogger(__name__)

class ECGDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed ECG windows and multi-hot labels.

    Windows are normalized at access time. Pass training-set mean and std
    for consistent normalization across train/val/test splits.
    Per-window normalization is used as fallback when stats are not provided.

    IMPORTANT: Always pass the SAME mean and std (computed from training data)
    to the val and test datasets. Never recompute on val/test — that is data leakage.

    Args:
        windows: np.ndarray of shape (N, window_size). Raw (unnormalized) ECG windows.
        labels: np.ndarray of shape (N, 5). Multi-hot float32 labels.
        mean: Training-set global mean. If None, per-window normalization is used.
        std: Training-set global std. If None, per-window normalization is used.
    """
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        mean: Optional[float] = None,
        std: Optional[float] = None
    ):
        if windows.ndim != 2:
            raise ValueError(f"windows must be 2D (N, window_size), got ndim={windows.ndim}")
        if windows.shape[0] != labels.shape[0]:
            raise ValueError(f"windows and labels must have same number of samples. Got windows={windows.shape[0]}, labels={labels.shape[0]}")
        if labels.shape[1] != 5:
            raise ValueError(f"labels must have 5 classes, got {labels.shape[1]}")

        self.windows = windows
        self.labels = labels
        self.mean = mean
        self.std = std

        logger.info(f"ECGDataset initialized: {len(windows)} samples, {labels.shape[1]} classes. Normalization: {'global stats' if mean is not None else 'per-window'}.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        if self.mean is not None and self.std is not None:
            normalized = (window - self.mean) / (self.std + 1e-8)
        else:
            normalized = (window - window.mean()) / (window.std() + 1e-8)
            
        x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
        y = torch.from_numpy(self.labels[idx].astype(np.float32))
        return x, y

def _make_dataloaders_base(
    config: dict,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    train_mean: float,
    train_std: float
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from preprocessed arrays.

    Normalization statistics must be computed from training data only and
    passed in — this function applies them uniformly to all splits.

    Args:
        config: Loaded config.yaml as dict.
        train_windows, val_windows, test_windows: Raw ECG window arrays, shape (N, window_size).
        train_labels, val_labels, test_labels: Multi-hot label arrays, shape (N, 5).
        train_mean: Mean computed from training windows ONLY.
        train_std: Std computed from training windows ONLY.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = ECGDataset(train_windows, train_labels, mean=train_mean, std=train_std)
    val_dataset = ECGDataset(val_windows, val_labels, mean=train_mean, std=train_std)
    test_dataset = ECGDataset(test_windows, test_labels, mean=train_mean, std=train_std)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config['training']['batch_size'] * 2,
        num_workers=config['training']['num_workers'],
        pin_memory=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config['training']['batch_size'] * 2,
        num_workers=config['training']['num_workers'],
        pin_memory=False,
        drop_last=False
    )

    logger.info(f"DataLoaders ready. Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} samples.")
    return train_loader, val_loader, test_loader

def make_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience wrapper to load X.npy and y.npy from disk, split them, 
    and return train/val/test DataLoaders.
    
    Args:
        config (dict): Parsed config.yaml.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    processed_dir = config['data']['processed_dir']
    X_path = os.path.join(os.getcwd(), processed_dir, "X.npy")
    y_path = os.path.join(os.getcwd(), processed_dir, "y.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        # Fallback for dry-run/testing: generate synthetic data if missing
        logger.warning(f"Data not found at {X_path}. Generating synthetic data for dry-run.")
        X = np.random.randn(100, 256) # Matching window_size from config
        y_indices = np.random.randint(0, 5, 100)
    else:
        X = np.load(X_path)
        y_indices = np.load(y_path)
        
    # Convert indices to multi-hot if needed
    if y_indices.ndim == 1:
        y = np.zeros((len(y_indices), 5))
        for i, val in enumerate(y_indices):
            if 0 <= int(val) < 5:
                y[i, int(val)] = 1
    else:
        y = y_indices
        
    # Simple split (70/15/15)
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_X, train_y = X[train_idx], y[train_idx]
    val_X, val_y = X[val_idx], y[val_idx]
    test_X, test_y = X[test_idx], y[test_idx]
    
    # Compute stats
    train_mean = float(np.mean(train_X))
    train_std = float(np.std(train_X))
    
    return _make_dataloaders_base(
        config, 
        train_X, train_y, 
        val_X, val_y, 
        test_X, test_y, 
        train_mean, train_std
    )
