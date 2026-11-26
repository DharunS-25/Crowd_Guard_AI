"""
dataset.py — CrowdGuard AI
===========================
PyTorch Dataset and DataLoader classes for temporal crowd sequence data.

Usage:
    dataset = CrowdSequenceDataset(X, y, meta)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

Each batch contains:
    x_batch : (batch, seq_len, n_features)  — float32
    y_batch : (batch,)                       — int64
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class CrowdSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-built temporal sequences.

    Parameters
    ----------
    X : np.ndarray of shape (N, seq_len, n_features)
    y : np.ndarray of shape (N,)
    meta : list of dicts with per-sequence metadata
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: Optional[List[dict]] = None,
    ):
        assert len(X) == len(y), "X and y must have equal length."
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.meta = meta or [{} for _ in range(len(y))]

        self.n_sequences = len(y)
        self.seq_len = X.shape[1]
        self.n_features = X.shape[2]

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def class_counts(self) -> Dict[int, int]:
        unique, counts = torch.unique(self.y, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def compute_class_weights(self, n_classes: int = 2) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for imbalanced datasets.

        Always returns a tensor of length `n_classes` even when some classes
        are absent from the data (absent classes get weight 0.0 so the loss
        contribution is zero for them, avoiding the CrossEntropyLoss shape
        mismatch that occurs when the weight tensor length != n_classes).
        """
        counts = self.class_counts()
        present_classes = set(counts.keys())
        # Clamp n_classes to be at least as large as max observed class + 1
        n_classes = max(n_classes, max(present_classes) + 1)

        if len(present_classes) < 2:
            logger.warning(
                f"Only {len(present_classes)} class(es) present in dataset "
                f"({present_classes}). Returning uniform weights — no class "
                "weighting applied."
            )
            return torch.ones(n_classes)

        total = sum(counts.values())
        weights = torch.zeros(n_classes)
        for cls, cnt in counts.items():
            weights[cls] = total / (len(present_classes) * cnt)
        logger.info(f"Class weights: {weights.tolist()}")
        return weights


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
    meta_train: Optional[List[dict]] = None,
    meta_val: Optional[List[dict]] = None,
    meta_test: Optional[List[dict]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, CrowdSequenceDataset]:
    """
    Create train, val, and test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, train_dataset
    """
    train_ds = CrowdSequenceDataset(X_train, y_train, meta_train)
    val_ds = CrowdSequenceDataset(X_val, y_val, meta_val)
    test_ds = CrowdSequenceDataset(X_test, y_test, meta_test)

    # Weighted sampling for class imbalance (training only)
    if use_weighted_sampler:
        counts = train_ds.class_counts()
        total = sum(counts.values())
        sample_weights = torch.zeros(len(train_ds))
        for cls, cnt in counts.items():
            idx = (train_ds.y == cls).nonzero(as_tuple=True)[0]
            sample_weights[idx] = total / (len(counts) * cnt)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info(
        f"DataLoaders — "
        f"train: {len(train_ds)} seqs, "
        f"val: {len(val_ds)} seqs, "
        f"test: {len(test_ds)} seqs"
    )
    return train_loader, val_loader, test_loader, train_ds
