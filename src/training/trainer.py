"""PyTorch Lightning training module and orchestrator.

Wraps model training in a LightningModule with configurable optimizers,
schedulers, and logging. The `train_model` function provides the high-level
entry point that wires everything together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers import MLFlowLogger

from src.data.datasets import CIFAR10DataModule
from src.models.cnn import CustomCNN
from src.models.transfer import BackboneType, TransferLearningModel
from src.training.callbacks import get_checkpoint_callback, get_early_stopping
from src.training.config import TrainingConfig


class ImageClassifier(pl.LightningModule):
    """Lightning module wrapping any classification model.

    Handles training/validation/test steps, metric computation,
    optimizer configuration, and LR scheduling.

    Args:
        model: The underlying PyTorch model.
        config: Training hyperparameters.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(config.to_dict())

        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self._shared_step(batch)
        self.train_acc(preds, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self._shared_step(batch)
        self.val_acc(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self._shared_step(batch)
        self.test_acc(preds, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        elif self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        else:
            return {"optimizer": optimizer}


def build_model(config: TrainingConfig) -> nn.Module:
    """Instantiate the appropriate model based on config.

    Args:
        config: Training configuration specifying model_name.

    Returns:
        Initialized model ready for training.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if config.model_name == "custom_cnn":
        return CustomCNN(
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    elif config.model_name == "resnet50":
        return TransferLearningModel(
            backbone=BackboneType.RESNET50,
            num_classes=config.num_classes,
            mode=config.mode,
        )
    elif config.model_name == "efficientnet_b0":
        return TransferLearningModel(
            backbone=BackboneType.EFFICIENTNET_B0,
            num_classes=config.num_classes,
            mode=config.mode,
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}")


def seed_everything(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    pl.seed_everything(seed, workers=True)


def train_model(config: TrainingConfig) -> pl.Trainer:
    """Execute full training pipeline.

    Creates the data module, model, logger, callbacks, and trainer,
    then runs training and testing.

    Args:
        config: Complete training configuration.

    Returns:
        The fitted Lightning Trainer instance.
    """
    seed_everything(config.seed)

    # Data
    use_imagenet = config.model_name in ("resnet50", "efficientnet_b0")
    data_module = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_fraction=config.val_fraction,
        image_size=config.image_size if use_imagenet else 32,
        use_imagenet_stats=use_imagenet,
        seed=config.seed,
    )

    # Model
    model = build_model(config)
    lit_model = ImageClassifier(model, config)

    # Logger
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    logger = MLFlowLogger(
        experiment_name=config.experiment_name,
        tracking_uri=f"file:{Path(config.output_dir).resolve() / 'mlruns'}",
    )

    # Callbacks
    callbacks = [
        get_early_stopping(config.early_stopping_patience),
        get_checkpoint_callback(config.output_dir),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        precision=config.precision,
        deterministic=True,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )

    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(lit_model, datamodule=data_module)

    return trainer
