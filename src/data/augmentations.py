"""Transform pipelines for training and evaluation.

Provides standardized augmentation strategies for CIFAR-10 and general image
classification tasks. Training transforms include standard data augmentation
(random crop, horizontal flip, color jitter) while test transforms only apply
normalization for deterministic evaluation.
"""

from __future__ import annotations

from torchvision import transforms

# CIFAR-10 channel-wise statistics (precomputed from training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# ImageNet statistics (used for transfer learning with pretrained models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 32,
    use_imagenet_stats: bool = False,
) -> transforms.Compose:
    """Build training augmentation pipeline.

    Args:
        image_size: Target spatial dimension (height = width).
        use_imagenet_stats: If True, normalize with ImageNet statistics
            (required for pretrained transfer learning models).

    Returns:
        Composed transform pipeline for training data.
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else CIFAR10_MEAN
    std = IMAGENET_STD if use_imagenet_stats else CIFAR10_STD

    transform_list = []

    if image_size != 32:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.extend([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(transform_list)


def get_test_transforms(
    image_size: int = 32,
    use_imagenet_stats: bool = False,
) -> transforms.Compose:
    """Build evaluation transform pipeline (no augmentation).

    Args:
        image_size: Target spatial dimension (height = width).
        use_imagenet_stats: If True, normalize with ImageNet statistics.

    Returns:
        Composed transform pipeline for evaluation data.
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else CIFAR10_MEAN
    std = IMAGENET_STD if use_imagenet_stats else CIFAR10_STD

    transform_list = []

    if image_size != 32:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(transform_list)
