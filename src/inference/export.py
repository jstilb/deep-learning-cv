"""ONNX model export for deployment.

Converts a trained PyTorch model to ONNX format with proper input/output
naming and dynamic batch size support. Includes validation to ensure
the exported model produces equivalent outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.training.config import TrainingConfig
from src.training.trainer import ImageClassifier, build_model


def export_to_onnx(
    checkpoint_path: str | Path,
    config: TrainingConfig,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export a trained model to ONNX format.

    Args:
        checkpoint_path: Path to the Lightning checkpoint.
        config: Training configuration.
        output_path: Where to save the .onnx file.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    model = build_model(config)
    lit_model = ImageClassifier.load_from_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        map_location="cpu",
    )
    lit_model.eval()

    image_size = config.image_size if config.model_name != "custom_cnn" else 32
    dummy_input = torch.randn(1, 3, image_size, image_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        lit_model.model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    return output_path


def validate_onnx(
    onnx_path: str | Path,
    checkpoint_path: str | Path,
    config: TrainingConfig,
    atol: float = 1e-5,
) -> bool:
    """Validate ONNX model produces equivalent outputs to PyTorch.

    Args:
        onnx_path: Path to the ONNX model.
        checkpoint_path: Path to the Lightning checkpoint.
        config: Training configuration.
        atol: Absolute tolerance for output comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import onnxruntime as ort

    # PyTorch inference
    model = build_model(config)
    lit_model = ImageClassifier.load_from_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        map_location="cpu",
    )
    lit_model.eval()

    image_size = config.image_size if config.model_name != "custom_cnn" else 32
    dummy_input = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        pytorch_output = lit_model.model(dummy_input).numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {"image": dummy_input.numpy()})

    return bool(np.allclose(pytorch_output, onnx_output[0], atol=atol))
