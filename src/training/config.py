"""Training hyperparameter configuration.

Centralizes all training hyperparameters with sensible defaults. Supports
loading from YAML config files for experiment reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Complete training configuration.

    Attributes:
        model_name: One of 'custom_cnn', 'resnet50', 'efficientnet_b0'.
        num_classes: Number of output classes.
        batch_size: Training batch size.
        max_epochs: Maximum training epochs.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        scheduler: LR scheduler type ('onecycle', 'cosine', 'none').
        dropout: Dropout probability.
        image_size: Input image spatial dimension.
        num_workers: DataLoader worker processes.
        val_fraction: Validation split fraction.
        early_stopping_patience: Epochs without improvement before stopping.
        seed: Global random seed.
        precision: Training precision ('32', '16-mixed').
        data_dir: Dataset storage directory.
        output_dir: Checkpoint and log output directory.
        experiment_name: MLflow experiment name.
        mode: Transfer learning mode ('feature_extraction', 'fine_tuning').
    """

    model_name: str = "custom_cnn"
    num_classes: int = 10
    batch_size: int = 128
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "onecycle"
    dropout: float = 0.1
    image_size: int = 32
    num_workers: int = 4
    val_fraction: float = 0.1
    early_stopping_patience: int = 10
    seed: int = 42
    precision: str = "32"
    data_dir: str = "./data"
    output_dir: str = "./experiments/results"
    experiment_name: str = "cifar10-classification"
    mode: str = "fine_tuning"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            TrainingConfig populated from file values.
        """
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
