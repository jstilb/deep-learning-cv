"""Tests for model architectures."""

from __future__ import annotations

import pytest
import torch

from src.models.cnn import ConvBlock, CustomCNN
from src.models.transfer import BackboneType, TransferLearningModel


class TestConvBlock:
    """Tests for the convolutional building block."""

    def test_output_shape(self) -> None:
        block = ConvBlock(3, 64)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_dropout_applied(self) -> None:
        block = ConvBlock(3, 64, dropout=0.5)
        block.train()
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)


class TestCustomCNN:
    """Tests for the custom CNN architecture."""

    def test_forward_shape(self) -> None:
        model = CustomCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_custom_num_classes(self) -> None:
        model = CustomCNN(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 100)

    def test_feature_extractor_returns_module(self) -> None:
        model = CustomCNN()
        features = model.get_feature_extractor()
        assert isinstance(features, torch.nn.Module)

    def test_single_channel_input(self) -> None:
        model = CustomCNN(in_channels=1)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_gradient_flow(self) -> None:
        model = CustomCNN()
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTransferLearningModel:
    """Tests for transfer learning wrappers."""

    @pytest.mark.slow
    def test_resnet50_forward_shape(self) -> None:
        model = TransferLearningModel(
            backbone=BackboneType.RESNET50,
            num_classes=10,
            pretrained=False,
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)

    @pytest.mark.slow
    def test_efficientnet_forward_shape(self) -> None:
        model = TransferLearningModel(
            backbone=BackboneType.EFFICIENTNET_B0,
            num_classes=10,
            pretrained=False,
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)

    def test_feature_extraction_freezes_backbone(self) -> None:
        model = TransferLearningModel(
            backbone=BackboneType.RESNET50,
            num_classes=10,
            mode="feature_extraction",
            pretrained=False,
        )
        # Most parameters should be frozen
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert frozen > total * 0.5

    def test_unfreeze_backbone(self) -> None:
        model = TransferLearningModel(
            backbone=BackboneType.RESNET50,
            num_classes=10,
            mode="feature_extraction",
            pretrained=False,
        )
        model.unfreeze_backbone()
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())
        assert trainable == total

    def test_parameter_counting(self) -> None:
        model = TransferLearningModel(
            backbone=BackboneType.RESNET50,
            num_classes=10,
            pretrained=False,
        )
        assert model.total_parameters() > 0
        assert model.trainable_parameters() > 0
        assert model.trainable_parameters() <= model.total_parameters()

    def test_target_layer_exists(self) -> None:
        model = TransferLearningModel(
            backbone=BackboneType.RESNET50,
            num_classes=10,
            pretrained=False,
        )
        layer = model.get_target_layer()
        assert isinstance(layer, torch.nn.Module)

    def test_string_backbone_name(self) -> None:
        model = TransferLearningModel(
            backbone="resnet50",
            num_classes=10,
            pretrained=False,
        )
        assert model.backbone_type == BackboneType.RESNET50
