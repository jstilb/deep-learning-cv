"""Transform pipelines for training and evaluation.

Provides standardized augmentation strategies for Food-101 and general image
classification tasks. Training transforms include standard data augmentation
(random crop, horizontal flip, color jitter) while test transforms only apply
normalization for deterministic evaluation.

Food-101 uses ImageNet normalization statistics (0.485, 0.456, 0.406) since
all models use ImageNet-pretrained backbones at 224x224 resolution.
"""

from __future__ import annotations

from torchvision import transforms

# ImageNet statistics — used for all models (Food-101 training with pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Food-101 normalization statistics (same as ImageNet for consistency)
FOOD101_MEAN = IMAGENET_MEAN
FOOD101_STD = IMAGENET_STD


def get_train_transforms(
    image_size: int = 224,
    use_imagenet_stats: bool = True,
) -> transforms.Compose:
    """Build training augmentation pipeline.

    Args:
        image_size: Target spatial dimension (height = width). Default 224 for Food-101.
        use_imagenet_stats: If True, normalize with ImageNet statistics (default).

    Returns:
        Composed transform pipeline for training data.
    """
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(transform_list)


def get_test_transforms(
    image_size: int = 224,
    use_imagenet_stats: bool = True,
) -> transforms.Compose:
    """Build evaluation transform pipeline (no augmentation).

    Args:
        image_size: Target spatial dimension (height = width). Default 224 for Food-101.
        use_imagenet_stats: If True, normalize with ImageNet statistics (default).

    Returns:
        Composed transform pipeline for evaluation data.
    """
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    transform_list = [
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(transform_list)
