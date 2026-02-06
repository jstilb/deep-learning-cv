# Architecture

## System Overview

This project implements an image classification pipeline with three model architectures, a unified training framework, comprehensive evaluation, and deployment-ready inference.

```mermaid
graph TB
    subgraph Data Pipeline
        A[CIFAR-10 Download] --> B[Augmentation Pipeline]
        B --> C[Stratified Split]
        C --> D[Train DataLoader]
        C --> E[Val DataLoader]
        C --> F[Test DataLoader]
    end

    subgraph Models
        G[Custom CNN<br/>5-block VGG-style]
        H[ResNet-50<br/>Transfer Learning]
        I[EfficientNet-B0<br/>Transfer Learning]
    end

    subgraph Training
        D --> J[PyTorch Lightning Trainer]
        E --> J
        J --> K[MLflow Tracking]
        J --> L[Early Stopping]
        J --> M[Checkpointing]
    end

    subgraph Evaluation
        F --> N[Metrics Suite]
        N --> O[Confusion Matrix]
        N --> P[ROC Curves]
        N --> Q[Error Analysis]
    end

    subgraph Deployment
        M --> R[ONNX Export]
        R --> S[Validated ONNX Model]
        M --> T[Grad-CAM Viz]
    end

    G --> J
    H --> J
    I --> J
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant DataModule
    participant Model
    participant Trainer
    participant MLflow

    User->>CLI: dlcv train --config custom_cnn.yaml
    CLI->>DataModule: CIFAR10DataModule(config)
    DataModule->>DataModule: Download + Split + Augment
    CLI->>Model: build_model(config)
    CLI->>Trainer: Trainer.fit(model, data)
    loop Each Epoch
        Trainer->>Model: training_step(batch)
        Model-->>Trainer: loss, accuracy
        Trainer->>MLflow: log_metrics()
        Trainer->>Model: validation_step(batch)
        Model-->>Trainer: val_loss, val_acc
    end
    Trainer->>Trainer: Early stopping check
    Trainer-->>User: Best checkpoint saved
```

## Model Architectures

### Custom CNN

A VGG-inspired architecture with 5 convolutional blocks, progressive channel widening (64 -> 512), BatchNorm after every convolution, and global average pooling before the classifier head.

```
Input (3x32x32)
  -> Conv(64) + BN + ReLU + MaxPool    -> 64x16x16
  -> Conv(128) + BN + ReLU + MaxPool   -> 128x8x8
  -> Conv(256)x2 + BN + ReLU + MaxPool -> 256x4x4
  -> Conv(512)x2 + BN + ReLU + MaxPool -> 512x2x2
  -> Conv(512)x2 + BN + ReLU           -> 512x2x2
  -> AdaptiveAvgPool(1x1)              -> 512
  -> FC(256) + ReLU + FC(10)           -> 10
```

### Transfer Learning Models

Both ResNet-50 and EfficientNet-B0 use ImageNet-pretrained backbones with a replaced classifier head. Two training modes are supported:

- **Feature Extraction**: Backbone frozen, only classifier trains (fast, lower accuracy)
- **Fine-Tuning**: Entire model trains with lower backbone LR (slower, higher accuracy)

## Key Design Decisions

See the [decisions](decisions/) directory for detailed ADRs:

- [001 - PyTorch over TensorFlow](decisions/001-pytorch-over-tensorflow.md)
- [002 - Transfer Learning Strategy](decisions/002-transfer-learning-strategy.md)
