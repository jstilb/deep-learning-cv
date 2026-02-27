"""
Food-101 Dataset — Real-World Image Classification
ISC-8504: Replace CIFAR-10 with a real-world dataset (Food-101)

Food-101 is a challenging real-world food image classification dataset with:
- 101 food categories
- 101,000 images (750 train + 250 test per class)
- High intra-class variance, realistic photography conditions

No CIFAR-10 data loading code in this module.

Reference: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
PyTorch: torchvision.datasets.Food101

Usage:
    dm = Food101DataModule(data_dir="./data", batch_size=64)
    dm.setup()
    train_loader = dm.train_dataloader()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import Food101

# Food-101 class list (101 categories, alphabetically sorted)
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles",
]

NUM_CLASSES = 101


def get_food101_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training augmentation pipeline for Food-101."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet statistics
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_food101_test_transforms(image_size: int = 224) -> transforms.Compose:
    """Test/validation transforms for Food-101 (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class Food101DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Food-101 real-world food classification.

    Replaces CIFAR-10 with a production-quality dataset featuring:
    - 101 food categories from restaurant photography
    - Realistic image quality variation and background clutter
    - Standard train/test split used in academic benchmarks

    Args:
        data_dir: Root directory for dataset storage.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        val_fraction: Fraction of training data held out for validation.
        image_size: Spatial dimension (default 224 for ImageNet-pretrained backbones).
        num_classes: Number of classes (use <101 for a subset, e.g., top-10 classes).
        seed: Reproducible split seed.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        image_size: int = 224,
        num_classes: int = NUM_CLASSES,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.image_size = image_size
        self.num_classes = min(num_classes, NUM_CLASSES)
        self.seed = seed

        self.train_transforms = get_food101_train_transforms(image_size)
        self.test_transforms = get_food101_test_transforms(image_size)

        self._train_dataset: Optional[Food101] = None
        self._val_dataset: Optional[Food101] = None
        self._test_dataset: Optional[Food101] = None

    def prepare_data(self) -> None:
        """Download Food-101 dataset (called once, on rank 0 only)."""
        Food101(root=str(self.data_dir), split="train", download=True)
        Food101(root=str(self.data_dir), split="test", download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test splits."""
        if stage in ("fit", None):
            full_train = Food101(
                root=str(self.data_dir),
                split="train",
                transform=self.train_transforms,
            )
            n_val = int(len(full_train) * self.val_fraction)
            n_train = len(full_train) - n_val
            self._train_dataset, self._val_dataset = random_split(
                full_train,
                [n_train, n_val],
                generator=__import__("torch").Generator().manual_seed(self.seed),
            )
            # Val uses test transforms (no augmentation)
            self._val_dataset.dataset.transform = self.test_transforms

        if stage in ("test", None):
            self._test_dataset = Food101(
                root=str(self.data_dir),
                split="test",
                transform=self.test_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "Call setup() first."
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup() first."
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None, "Call setup(stage='test') first."
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def classes(self) -> list[str]:
        """Return list of class names."""
        return FOOD101_CLASSES[: self.num_classes]

    def __repr__(self) -> str:
        return (
            f"Food101DataModule("
            f"num_classes={self.num_classes}, "
            f"batch_size={self.batch_size}, "
            f"image_size={self.image_size})"
        )
