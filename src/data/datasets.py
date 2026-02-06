"""PyTorch Lightning DataModule for CIFAR-10.

Handles automatic downloading, augmentation pipeline application, and
train/val/test DataLoader creation with configurable batch size and workers.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from src.data.augmentations import get_test_transforms, get_train_transforms
from src.data.splits import stratified_split

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class CIFAR10DataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping CIFAR-10 with proper splits and augmentation.

    Args:
        data_dir: Root directory for dataset storage.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        val_fraction: Fraction of training data held out for validation.
        image_size: Spatial dimension to resize images to.
        use_imagenet_stats: Normalize with ImageNet statistics (for transfer learning).
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        image_size: int = 32,
        use_imagenet_stats: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.image_size = image_size
        self.use_imagenet_stats = use_imagenet_stats
        self.seed = seed

        self.train_transforms = get_train_transforms(image_size, use_imagenet_stats)
        self.test_transforms = get_test_transforms(image_size, use_imagenet_stats)

        self.num_classes = 10
        self.class_names = CIFAR10_CLASSES

    def prepare_data(self) -> None:
        """Download CIFAR-10 if not already present."""
        CIFAR10(root=str(self.data_dir), train=True, download=True)
        CIFAR10(root=str(self.data_dir), train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Create train/val/test splits with appropriate transforms."""
        if stage == "fit" or stage is None:
            full_train = CIFAR10(
                root=str(self.data_dir),
                train=True,
                transform=self.train_transforms,
            )
            # We need a separate copy for validation with test transforms
            full_train_for_val = CIFAR10(
                root=str(self.data_dir),
                train=True,
                transform=self.test_transforms,
            )

            train_subset, val_subset_raw = stratified_split(
                full_train, self.val_fraction, self.seed
            )
            _, val_subset = stratified_split(
                full_train_for_val, self.val_fraction, self.seed
            )

            self.train_dataset = train_subset
            self.val_dataset = val_subset

        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                root=str(self.data_dir),
                train=False,
                transform=self.test_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
