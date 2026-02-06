# ADR 002: Transfer Learning Strategy

## Status

Accepted

## Context

CIFAR-10 images are 32x32 pixels -- significantly smaller than the 224x224 inputs that ImageNet-pretrained models expect. We need to decide how to handle this mismatch and which transfer learning approach to use.

## Decision

We resize CIFAR-10 images to 224x224 for transfer learning models and use **full fine-tuning** as the primary training mode, with **feature extraction** available for comparison.

## Rationale

### Resizing Strategy

1. **Upsampling to 224x224**: While upsampling introduces interpolation artifacts, pretrained models have learned rich hierarchical features from ImageNet that transfer well even with upsampled inputs. The alternative (modifying early layers for 32x32) breaks the pretrained weight structure.

2. **Performance precedent**: Multiple papers demonstrate that even with resizing, transfer learning from ImageNet significantly outperforms training from scratch on small datasets.

### Fine-Tuning vs Feature Extraction

1. **Fine-tuning (selected default)**: Allows the backbone to adapt its learned features to the CIFAR-10 domain. Lower learning rate for backbone prevents catastrophic forgetting while enabling adaptation.

2. **Feature extraction (comparison mode)**: Freezes backbone, trains only the classifier head. Faster but typically 5-10% lower accuracy on CIFAR-10. Included for educational comparison.

### Model Selection

- **ResNet-50**: The "standard" transfer learning baseline. Well-understood architecture, reliable performance, fast inference. 25.6M parameters.
- **EfficientNet-B0**: Modern compound-scaled architecture with better accuracy-per-FLOP. Demonstrates awareness of efficiency-focused architectures. 5.3M parameters.

## Consequences

- Transfer learning models require ~7x more memory due to 224x224 inputs (vs 32x32).
- Training is slower per epoch but converges in fewer epochs.
- ImageNet normalization statistics must be used (not CIFAR-10 statistics).
- Need separate augmentation pipelines for 32x32 and 224x224 inputs.

## Metrics

Expected accuracy ranges (from literature):

| Model | Expected Accuracy |
|-------|------------------|
| Custom CNN (from scratch) | 88-92% |
| ResNet-50 (fine-tuned) | 95-97% |
| EfficientNet-B0 (fine-tuned) | 95-97% |
| ResNet-50 (feature extraction) | 85-90% |
