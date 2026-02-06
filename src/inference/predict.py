"""Single-image and batch inference.

Loads a trained model checkpoint and runs predictions on individual images
or directories of images with progress reporting.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from rich.progress import track
from torchvision import transforms

from src.data.augmentations import get_test_transforms
from src.data.datasets import CIFAR10_CLASSES
from src.training.config import TrainingConfig
from src.training.trainer import ImageClassifier, build_model


def load_model(
    checkpoint_path: str | Path,
    config: TrainingConfig,
    device: str = "cpu",
) -> ImageClassifier:
    """Load a trained model from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.
        config: Training configuration used to build the model.
        device: Target device ('cpu' or 'cuda').

    Returns:
        Model loaded and set to evaluation mode.
    """
    model = build_model(config)
    lit_model = ImageClassifier.load_from_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        map_location=device,
    )
    lit_model.eval()
    lit_model.to(device)
    return lit_model


def predict_single(
    model: ImageClassifier,
    image_path: str | Path,
    image_size: int = 32,
    use_imagenet_stats: bool = False,
    device: str = "cpu",
) -> dict:
    """Run prediction on a single image.

    Args:
        model: Loaded model in eval mode.
        image_path: Path to the input image.
        image_size: Expected input size.
        use_imagenet_stats: Whether to use ImageNet normalization.
        device: Compute device.

    Returns:
        Dict with 'class_name', 'class_index', 'confidence', and 'probabilities'.
    """
    transform = get_test_transforms(image_size, use_imagenet_stats)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    confidence, pred_idx = torch.max(probs, dim=1)

    return {
        "class_name": CIFAR10_CLASSES[pred_idx.item()],
        "class_index": pred_idx.item(),
        "confidence": confidence.item(),
        "probabilities": {
            CIFAR10_CLASSES[i]: float(probs[0][i]) for i in range(len(CIFAR10_CLASSES))
        },
    }


def predict_batch(
    model: ImageClassifier,
    image_dir: str | Path,
    image_size: int = 32,
    use_imagenet_stats: bool = False,
    device: str = "cpu",
) -> list[dict]:
    """Run predictions on all images in a directory.

    Args:
        model: Loaded model in eval mode.
        image_dir: Directory containing images.
        image_size: Expected input size.
        use_imagenet_stats: Whether to use ImageNet normalization.
        device: Compute device.

    Returns:
        List of prediction dicts, one per image.
    """
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in extensions
    )

    results = []
    for path in track(image_paths, description="Running inference..."):
        result = predict_single(model, path, image_size, use_imagenet_stats, device)
        result["file"] = str(path)
        results.append(result)

    return results
