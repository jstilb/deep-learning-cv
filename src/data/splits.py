"""Stratified train/validation/test split utilities.

Ensures class distribution is preserved across all splits, which is critical
for reliable evaluation on imbalanced datasets.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from torch.utils.data import Dataset, Subset


def stratified_split(
    dataset: Dataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split a dataset into train and validation subsets with stratified sampling.

    Preserves the class distribution from the original dataset in both
    the training and validation splits.

    Args:
        dataset: Source dataset with a `targets` attribute (e.g., CIFAR-10).
        val_fraction: Fraction of samples to use for validation (0 < val_fraction < 1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_subset, val_subset).

    Raises:
        ValueError: If val_fraction is not in (0, 1).
        AttributeError: If dataset has no `targets` attribute.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    targets: Sequence[int] = dataset.targets  # type: ignore[attr-defined]
    targets_array = np.array(targets)
    classes = np.unique(targets_array)

    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in classes:
        cls_indices = np.where(targets_array == cls)[0]
        rng.shuffle(cls_indices)

        n_val = max(1, int(len(cls_indices) * val_fraction))
        val_indices.extend(cls_indices[:n_val].tolist())
        train_indices.extend(cls_indices[n_val:].tolist())

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
