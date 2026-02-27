"""Command-line interface for training, evaluation, and inference.

Usage:
    dlcv train --config experiments/configs/custom_cnn.yaml
    dlcv evaluate --checkpoint experiments/results/best.ckpt --model custom_cnn
    dlcv predict --checkpoint experiments/results/best.ckpt --image photo.jpg
    dlcv export --checkpoint experiments/results/best.ckpt --output model.onnx
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="dlcv",
    help="Deep Learning Computer Vision -- Food-101 classification with CNNs, transfer learning, and CLIP zero-shot.",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    config: str = typer.Option(
        None, "--config", "-c", help="Path to YAML config file."
    ),
    model: str = typer.Option(
        "custom_cnn", "--model", "-m", help="Model name: custom_cnn, resnet50, efficientnet_b0."
    ),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Max training epochs."),
    batch_size: int = typer.Option(128, "--batch-size", "-b", help="Batch size."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    output_dir: str = typer.Option("./experiments/results", "--output-dir", "-o"),
) -> None:
    """Train a model on Food-101."""
    from src.training.config import TrainingConfig
    from src.training.trainer import train_model

    if config:
        cfg = TrainingConfig.from_yaml(config)
    else:
        image_size = 224 if model in ("resnet50", "efficientnet_b0") else 32
        cfg = TrainingConfig(
            model_name=model,
            max_epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=seed,
            output_dir=output_dir,
            image_size=image_size,
        )

    console.print(f"[bold green]Training {cfg.model_name}[/bold green]")
    console.print(f"  Epochs: {cfg.max_epochs}")
    console.print(f"  Batch size: {cfg.batch_size}")
    console.print(f"  Learning rate: {cfg.learning_rate}")
    console.print(f"  Image size: {cfg.image_size}")
    console.print()

    train_model(cfg)
    console.print("[bold green]Training complete![/bold green]")


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(help="Path to model checkpoint."),
    model: str = typer.Option(
        "custom_cnn", "--model", "-m", help="Model architecture."
    ),
    output_dir: str = typer.Option("./experiments/results", "--output-dir", "-o"),
) -> None:
    """Evaluate a trained model and generate metrics/plots."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from src.data.food101_dataset import FOOD101_CLASSES, Food101DataModule
    from src.evaluation.analysis import (
        find_confident_errors,
        plot_confusion_matrix,
        plot_roc_curves,
    )
    from src.evaluation.metrics import compute_metrics
    from src.inference.predict import load_model
    from src.training.config import TrainingConfig

    # Food-101 requires 224x224 for ImageNet-pretrained backbones
    image_size = 224

    cfg = TrainingConfig(
        model_name=model, image_size=image_size, num_workers=0
    )
    lit_model = load_model(checkpoint, cfg, device="cpu")

    dm = Food101DataModule(
        batch_size=64,
        num_workers=0,
        image_size=image_size,
    )
    dm.prepare_data()
    dm.setup(stage="test")

    all_preds, all_targets, all_probs = [], [], []
    for batch in dm.test_dataloader():
        images, targets = batch
        with torch.no_grad():
            logits = lit_model(images)
            probs = F.softmax(logits, dim=1)
        all_preds.append(torch.argmax(logits, dim=1))
        all_targets.append(targets)
        all_probs.append(probs)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_probs).numpy()

    results = compute_metrics(
        y_true, y_pred, y_prob,
        class_names=list(FOOD101_CLASSES),
    )

    # Print results table
    table = Table(title="Per-Class Metrics")
    table.add_column("Class", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")

    for name in FOOD101_CLASSES:
        if name in results.per_class_report:
            r = results.per_class_report[name]
            table.add_row(
                name,
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1-score']:.4f}",
            )

    console.print(table)
    console.print(f"\n[bold]Overall Accuracy:[/bold] {results.accuracy:.4f}")
    if results.roc_auc:
        console.print(f"[bold]Macro ROC AUC:[/bold] {results.roc_auc:.4f}")

    # Save plots
    output = Path(output_dir)
    plot_confusion_matrix(results, list(FOOD101_CLASSES), output / "confusion_matrix.png")
    if results.per_class_roc:
        plot_roc_curves(results, list(FOOD101_CLASSES), output / "roc_curves.png")

    # Save errors
    errors = find_confident_errors(y_true, y_pred, y_prob)
    if errors:
        with open(output / "confident_errors.json", "w") as f:
            json.dump(errors, f, indent=2)

    console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def predict(
    image: str = typer.Argument(help="Path to image or directory."),
    checkpoint: str = typer.Option(
        ..., "--checkpoint", "-c", help="Path to model checkpoint."
    ),
    model: str = typer.Option(
        "custom_cnn", "--model", "-m", help="Model architecture."
    ),
) -> None:
    """Run inference on a single image or directory."""
    from src.inference.predict import load_model, predict_batch, predict_single
    from src.training.config import TrainingConfig

    use_imagenet = model in ("resnet50", "efficientnet_b0")
    image_size = 224 if use_imagenet else 32

    cfg = TrainingConfig(model_name=model, image_size=image_size)
    lit_model = load_model(checkpoint, cfg, device="cpu")

    path = Path(image)
    if path.is_dir():
        results = predict_batch(lit_model, path, image_size, use_imagenet)
        for r in results:
            console.print(
                f"  {Path(r['file']).name}: "
                f"[bold]{r['class_name']}[/bold] ({r['confidence']:.2%})"
            )
    else:
        result = predict_single(lit_model, path, image_size, use_imagenet)
        console.print(f"[bold]Prediction:[/bold] {result['class_name']}")
        console.print(f"[bold]Confidence:[/bold] {result['confidence']:.2%}")

        table = Table(title="Class Probabilities")
        table.add_column("Class", style="cyan")
        table.add_column("Probability", justify="right")
        for name, prob in sorted(
            result["probabilities"].items(), key=lambda x: -x[1]
        ):
            table.add_row(name, f"{prob:.4f}")
        console.print(table)


@app.command()
def export(
    checkpoint: str = typer.Argument(help="Path to model checkpoint."),
    output: str = typer.Option(
        "./model.onnx", "--output", "-o", help="Output ONNX file path."
    ),
    model: str = typer.Option(
        "custom_cnn", "--model", "-m", help="Model architecture."
    ),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate exported model."
    ),
) -> None:
    """Export a trained model to ONNX format."""
    from src.inference.export import export_to_onnx, validate_onnx
    from src.training.config import TrainingConfig

    use_imagenet = model in ("resnet50", "efficientnet_b0")
    image_size = 224 if use_imagenet else 32

    cfg = TrainingConfig(model_name=model, image_size=image_size)

    console.print(f"[bold]Exporting {model} to ONNX...[/bold]")
    onnx_path = export_to_onnx(checkpoint, cfg, output)
    console.print(f"[green]Saved to {onnx_path}[/green]")

    if validate:
        console.print("Validating ONNX output equivalence...")
        is_valid = validate_onnx(onnx_path, checkpoint, cfg)
        if is_valid:
            console.print("[bold green]Validation passed![/bold green]")
        else:
            console.print("[bold red]Validation FAILED -- outputs differ![/bold red]")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
