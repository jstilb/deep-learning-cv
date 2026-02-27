"""Grad-CAM visualization for model interpretability.

Generates class activation maps showing which image regions most influenced
the model's prediction. Uses the pytorch-grad-cam library under the hood,
with convenience wrappers for our model architectures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.data.augmentations import IMAGENET_MEAN, IMAGENET_STD

# Food-101 uses ImageNet normalization statistics
# (same as IMAGENET_MEAN/IMAGENET_STD — kept for legacy API compatibility)
FOOD101_MEAN = IMAGENET_MEAN
FOOD101_STD = IMAGENET_STD


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> np.ndarray:
    """Convert a normalized tensor back to a displayable image array.

    Args:
        tensor: Normalized image tensor of shape (C, H, W).
        mean: Channel-wise mean used during normalization.
        std: Channel-wise std used during normalization.

    Returns:
        NumPy array of shape (H, W, C) with values in [0, 1].
    """
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def generate_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    use_imagenet_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM heatmap for a single input image.

    Args:
        model: Trained classification model.
        target_layer: Convolutional layer to visualize.
        input_tensor: Preprocessed image tensor of shape (1, C, H, W).
        target_class: Class index to visualize. None = predicted class.
        use_imagenet_stats: Whether ImageNet normalization was used.

    Returns:
        Tuple of (cam_image, grayscale_cam) where cam_image is the
        overlay visualization and grayscale_cam is the raw heatmap.
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else IMAGENET_MEAN
    std = IMAGENET_STD if use_imagenet_stats else IMAGENET_STD

    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = None
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Denormalize the input image for overlay
    rgb_img = denormalize(input_tensor[0], mean, std)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image, grayscale_cam


def save_gradcam_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: str | Path,
    cols: int = 4,
) -> None:
    """Save a grid of Grad-CAM visualizations to disk.

    Args:
        images: List of cam overlay images (H, W, 3) uint8.
        titles: Title for each image.
        output_path: File path to save the figure.
        cols: Number of columns in the grid.
    """
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for idx, (img, title) in enumerate(zip(images, titles)):
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis("off")

    # Hide unused subplot axes
    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
