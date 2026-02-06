"""PyTorch Lightning callbacks for training control.

Provides early stopping and model checkpointing configured
for image classification tasks.
"""

from __future__ import annotations

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_early_stopping(patience: int = 10) -> EarlyStopping:
    """Create an early stopping callback monitoring validation loss.

    Args:
        patience: Number of epochs without improvement before stopping.

    Returns:
        Configured EarlyStopping callback.
    """
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True,
    )


def get_checkpoint_callback(output_dir: str = "./experiments/results") -> ModelCheckpoint:
    """Create a model checkpoint callback saving the best model by validation accuracy.

    Args:
        output_dir: Directory to store checkpoint files.

    Returns:
        Configured ModelCheckpoint callback.
    """
    return ModelCheckpoint(
        dirpath=output_dir,
        filename="best-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
