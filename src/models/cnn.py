"""Custom CNN architecture for Food-101 classification.

A 5-block convolutional network with BatchNorm, ReLU, and Dropout -- designed
as a strong baseline before comparing against transfer learning approaches.
Architecture inspired by VGG-style progressive filter widening.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> optional Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class CustomCNN(nn.Module):
    """5-block CNN with progressive channel widening and global average pooling.

    Architecture:
        Block 1: 3 -> 64 channels, MaxPool
        Block 2: 64 -> 128 channels, MaxPool
        Block 3: 128 -> 256 channels (x2 convs), MaxPool
        Block 4: 256 -> 512 channels (x2 convs), MaxPool
        Block 5: 512 -> 512 channels (x2 convs), MaxPool
        Global Average Pooling -> FC -> num_classes

    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability applied after each conv block.
        in_channels: Number of input image channels (3 for RGB).
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.1,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            ConvBlock(in_channels, 64, dropout=dropout),
            nn.MaxPool2d(2, 2),
            # Block 2
            ConvBlock(64, 128, dropout=dropout),
            nn.MaxPool2d(2, 2),
            # Block 3 (double conv)
            ConvBlock(128, 256, dropout=dropout),
            ConvBlock(256, 256, dropout=dropout),
            nn.MaxPool2d(2, 2),
            # Block 4 (double conv)
            ConvBlock(256, 512, dropout=dropout),
            ConvBlock(512, 512, dropout=dropout),
            nn.MaxPool2d(2, 2),
            # Block 5 (double conv)
            ConvBlock(512, 512, dropout=dropout),
            ConvBlock(512, 512, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout * 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_feature_extractor(self) -> nn.Module:
        """Return the convolutional feature extractor (useful for Grad-CAM)."""
        return self.features
