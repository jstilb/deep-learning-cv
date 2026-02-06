"""Transfer learning wrappers for pretrained models.

Supports ResNet-50 and EfficientNet-B0 with two modes:
  - Feature extraction: freeze backbone, train only the classifier head
  - Fine-tuning: unfreeze backbone (optionally after N epochs), train end-to-end

Both modes replace the final classification layer to match the target number of classes.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


class BackboneType(str, Enum):
    """Supported pretrained backbone architectures."""

    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"


class TransferLearningModel(nn.Module):
    """Transfer learning model with configurable backbone and training mode.

    Args:
        backbone: Which pretrained architecture to use.
        num_classes: Number of output classes.
        mode: 'feature_extraction' freezes the backbone; 'fine_tuning' trains everything.
        pretrained: Load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        backbone: BackboneType | str = BackboneType.RESNET50,
        num_classes: int = 10,
        mode: Literal["feature_extraction", "fine_tuning"] = "fine_tuning",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_type = BackboneType(backbone)
        self.num_classes = num_classes
        self.mode = mode

        if self.backbone_type == BackboneType.RESNET50:
            self._build_resnet(num_classes, pretrained)
        elif self.backbone_type == BackboneType.EFFICIENTNET_B0:
            self._build_efficientnet(num_classes, pretrained)

        if mode == "feature_extraction":
            self._freeze_backbone()

    def _build_resnet(self, num_classes: int, pretrained: bool) -> None:
        """Initialize ResNet-50 with a new classifier head."""
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        # Expose the last conv layer for Grad-CAM
        self._target_layer = self.backbone.layer4[-1]

    def _build_efficientnet(self, num_classes: int, pretrained: bool) -> None:
        """Initialize EfficientNet-B0 with a new classifier head."""
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        self._target_layer = self.backbone.features[-1]

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (call after initial training epochs)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.mode = "fine_tuning"

    def get_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        return self._target_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
