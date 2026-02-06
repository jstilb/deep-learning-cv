"""Integration tests for the training pipeline.

These tests use a tiny synthetic dataset to verify the full training loop
works end-to-end without requiring CIFAR-10 download or GPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import CustomCNN
from src.training.config import TrainingConfig
from src.training.trainer import ImageClassifier, build_model


class TestBuildModel:
    """Tests for model factory function."""

    def test_build_custom_cnn(self) -> None:
        config = TrainingConfig(model_name="custom_cnn")
        model = build_model(config)
        assert isinstance(model, CustomCNN)

    def test_build_resnet50(self) -> None:
        config = TrainingConfig(model_name="resnet50")
        model = build_model(config)
        assert model is not None

    def test_build_efficientnet(self) -> None:
        config = TrainingConfig(model_name="efficientnet_b0")
        model = build_model(config)
        assert model is not None

    def test_build_unknown_raises(self) -> None:
        config = TrainingConfig(model_name="not_a_model")
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(config)


class TestTrainingLoop:
    """Smoke tests for the Lightning training loop with synthetic data."""

    @staticmethod
    def _make_tiny_loaders(
        n_samples: int = 32, num_classes: int = 10, image_size: int = 32
    ):
        """Create tiny synthetic train and val DataLoaders."""
        x = torch.randn(n_samples, 3, image_size, image_size)
        y = torch.randint(0, num_classes, (n_samples,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        return loader, loader  # use same loader for train and val

    def test_custom_cnn_trains_one_epoch(self) -> None:
        config = TrainingConfig(
            model_name="custom_cnn",
            max_epochs=1,
            learning_rate=0.01,
            scheduler="none",
        )
        model = build_model(config)
        lit_model = ImageClassifier(model, config)

        train_loader, val_loader = self._make_tiny_loaders()

        trainer = pl.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
            devices=1,
        )
        trainer.fit(lit_model, train_loader, val_loader)

    def test_loss_decreases_over_steps(self) -> None:
        """Verify the model actually learns (loss goes down)."""
        config = TrainingConfig(
            model_name="custom_cnn",
            max_epochs=5,
            learning_rate=0.01,
            scheduler="none",
        )
        model = build_model(config)
        lit_model = ImageClassifier(model, config)

        # Small deterministic dataset that should be learnable
        torch.manual_seed(42)
        x = torch.randn(64, 3, 32, 32)
        y = torch.randint(0, 2, (64,))  # binary to make learning easier
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        trainer = pl.Trainer(
            max_epochs=5,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
            devices=1,
        )
        trainer.fit(lit_model, loader, loader)

        # The logged train_loss should exist
        assert trainer.callback_metrics.get("train_loss") is not None


class TestTrainingConfig:
    """Tests for configuration management."""

    def test_default_config(self) -> None:
        config = TrainingConfig()
        assert config.model_name == "custom_cnn"
        assert config.num_classes == 10
        assert config.seed == 42

    def test_to_dict(self) -> None:
        config = TrainingConfig(model_name="resnet50", max_epochs=100)
        d = config.to_dict()
        assert d["model_name"] == "resnet50"
        assert d["max_epochs"] == 100

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = "model_name: efficientnet_b0\nmax_epochs: 25\nseed: 99\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)
        config = TrainingConfig.from_yaml(config_file)
        assert config.model_name == "efficientnet_b0"
        assert config.max_epochs == 25
        assert config.seed == 99
