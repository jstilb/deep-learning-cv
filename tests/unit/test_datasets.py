"""Tests for data pipeline components."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision import transforms

from src.data.augmentations import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_test_transforms,
    get_train_transforms,
)
from src.data.splits import stratified_split


class TestAugmentations:
    """Tests for transform pipeline construction."""

    def test_train_transforms_returns_compose(self) -> None:
        t = get_train_transforms()
        assert isinstance(t, transforms.Compose)

    def test_test_transforms_returns_compose(self) -> None:
        t = get_test_transforms()
        assert isinstance(t, transforms.Compose)

    def test_train_transforms_include_augmentation(self) -> None:
        t = get_train_transforms()
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "RandomHorizontalFlip" in transform_types
        assert "ColorJitter" in transform_types

    def test_test_transforms_no_augmentation(self) -> None:
        t = get_test_transforms()
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "RandomHorizontalFlip" not in transform_types
        assert "ColorJitter" not in transform_types

    def test_imagenet_stats_flag(self) -> None:
        t = get_train_transforms(use_imagenet_stats=True)
        # Find the Normalize transform and check its values
        normalize = None
        for tr in t.transforms:
            if isinstance(tr, transforms.Normalize):
                normalize = tr
                break
        assert normalize is not None
        np.testing.assert_allclose(normalize.mean, IMAGENET_MEAN, atol=1e-4)
        np.testing.assert_allclose(normalize.std, IMAGENET_STD, atol=1e-4)

    def test_food101_stats_default(self) -> None:
        """Food-101 uses ImageNet stats (same as IMAGENET_MEAN/STD)."""
        t = get_train_transforms(use_imagenet_stats=True)
        normalize = None
        for tr in t.transforms:
            if isinstance(tr, transforms.Normalize):
                normalize = tr
                break
        assert normalize is not None
        np.testing.assert_allclose(normalize.mean, IMAGENET_MEAN, atol=1e-4)

    def test_output_is_tensor(self) -> None:
        t = get_test_transforms()
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = t(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # 3 channels


class TestStratifiedSplit:
    """Tests for stratified train/val splitting."""

    def _make_fake_dataset(self, n_per_class: int = 100, n_classes: int = 10):
        """Create a minimal dataset-like object with targets."""

        class FakeDataset:
            def __init__(self, targets):
                self.targets = targets
                self.data = list(range(len(targets)))

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

        targets = []
        for c in range(n_classes):
            targets.extend([c] * n_per_class)
        return FakeDataset(targets)

    def test_split_preserves_total_size(self) -> None:
        ds = self._make_fake_dataset()
        train_sub, val_sub = stratified_split(ds, val_fraction=0.1)
        assert len(train_sub) + len(val_sub) == len(ds)

    def test_split_class_distribution(self) -> None:
        ds = self._make_fake_dataset(n_per_class=100, n_classes=5)
        _, val_sub = stratified_split(ds, val_fraction=0.2)

        val_targets = [ds.targets[i] for i in val_sub.indices]
        from collections import Counter
        counts = Counter(val_targets)
        # Each class should have ~20 samples (20% of 100)
        for c in range(5):
            assert 15 <= counts[c] <= 25

    def test_split_deterministic(self) -> None:
        ds = self._make_fake_dataset()
        train1, val1 = stratified_split(ds, seed=42)
        train2, val2 = stratified_split(ds, seed=42)
        assert train1.indices == train2.indices
        assert val1.indices == val2.indices

    def test_split_different_seeds_differ(self) -> None:
        ds = self._make_fake_dataset()
        train1, _ = stratified_split(ds, seed=42)
        train2, _ = stratified_split(ds, seed=99)
        assert train1.indices != train2.indices

    def test_invalid_val_fraction_raises(self) -> None:
        ds = self._make_fake_dataset()
        with pytest.raises(ValueError):
            stratified_split(ds, val_fraction=0.0)
        with pytest.raises(ValueError):
            stratified_split(ds, val_fraction=1.0)
        with pytest.raises(ValueError):
            stratified_split(ds, val_fraction=-0.1)
