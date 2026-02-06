"""Error analysis and visualization tools.

Generates confusion matrix heatmaps, ROC curves, and identifies the
most confident incorrect predictions for debugging model behavior.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.metrics import ClassificationResults


def plot_confusion_matrix(
    results: ClassificationResults,
    class_names: list[str],
    output_path: str | Path,
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        results: Classification results containing the confusion matrix.
        class_names: List of class display names.
        output_path: File path to save the figure.
        normalize: If True, normalize rows to show percentages.
    """
    cm = results.confusion_matrix.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    results: ClassificationResults,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Plot and save per-class ROC curves.

    Args:
        results: Classification results containing ROC data.
        class_names: List of class display names.
        output_path: File path to save the figure.
    """
    if results.per_class_roc is None:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        if i in results.per_class_roc:
            fpr, tpr, _ = results.per_class_roc[i]
            ax.plot(fpr, tpr, label=f"{name}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves (Macro AUC = {results.roc_auc:.4f})", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def find_confident_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    """Find the most confident incorrect predictions.

    These are the samples where the model was most confident in its (wrong)
    prediction -- useful for understanding systematic failure modes.

    Args:
        y_true: Ground truth labels of shape (N,).
        y_pred: Predicted labels of shape (N,).
        y_prob: Prediction probabilities of shape (N, num_classes).
        top_k: Number of errors to return.

    Returns:
        List of dicts with keys: index, true_label, pred_label, confidence.
    """
    wrong_mask = y_true != y_pred
    if not np.any(wrong_mask):
        return []

    wrong_indices = np.where(wrong_mask)[0]
    confidences = np.max(y_prob[wrong_indices], axis=1)

    sorted_idx = np.argsort(-confidences)[:top_k]
    selected = wrong_indices[sorted_idx]

    errors = []
    for idx in selected:
        errors.append({
            "index": int(idx),
            "true_label": int(y_true[idx]),
            "pred_label": int(y_pred[idx]),
            "confidence": float(np.max(y_prob[idx])),
        })

    return errors
