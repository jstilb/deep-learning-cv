# ADR 001: PyTorch over TensorFlow

## Status

Accepted

## Context

We need a deep learning framework for building and training image classification models. The two dominant options are PyTorch and TensorFlow/Keras.

## Decision

We chose **PyTorch 2.x** as the framework for this project.

## Rationale

1. **Research ecosystem**: PyTorch dominates ML research. Most state-of-the-art papers release PyTorch implementations first, making it easier to adapt new techniques.

2. **Pythonic API**: PyTorch's eager execution model and Python-native design make debugging straightforward. `print()` and `pdb` work exactly as expected -- no graph compilation surprises.

3. **PyTorch Lightning**: Reduces boilerplate while preserving full control over the training loop. Lightning handles distributed training, mixed precision, logging, and checkpointing without framework lock-in.

4. **torch.compile (2.x)**: PyTorch 2.0's compiler provides TensorFlow-level performance without sacrificing the eager-mode development experience.

5. **ONNX ecosystem**: PyTorch has first-class ONNX export support, enabling deployment to any inference runtime (TensorRT, ONNX Runtime, OpenVINO).

6. **Industry adoption**: PyTorch has surpassed TensorFlow in industry adoption (2023 Stack Overflow survey, papers-with-code statistics). Skills are more transferable.

## Consequences

- Must use `torchvision` for pretrained models (not `timm`, though `timm` could be added later).
- MLflow integration is clean via `pytorch_lightning.loggers.MLFlowLogger`.
- No Keras-style high-level API, but Lightning fills this gap.
