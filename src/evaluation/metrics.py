"""Classification metrics computation.

Provides functions for computing per-class precision, recall, F1, confusion
matrices, and ROC curves for multi-class classification problems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


@dataclass
class ClassificationResults:
    """Container for classification evaluation results.

    Attributes:
        accuracy: Overall accuracy.
        per_class_report: Sklearn classification report as a dictionary.
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes).
        roc_auc: Macro-averaged ROC AUC score.
        per_class_roc: Dict mapping class index to (fpr, tpr, thresholds).
    """

    accuracy: float
    per_class_report: dict
    confusion_matrix: np.ndarray
    roc_auc: float | None
    per_class_roc: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None


def compute_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor | None = None,
    class_names: list[str] | None = None,
    num_classes: int = 10,
) -> ClassificationResults:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels of shape (N,).
        y_pred: Predicted labels of shape (N,).
        y_prob: Predicted probabilities of shape (N, num_classes). Optional,
            required for ROC/AUC computation.
        class_names: Optional list of class names for the report.
        num_classes: Number of classes.

    Returns:
        ClassificationResults with all computed metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    accuracy = float(np.mean(y_true == y_pred))

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    roc_auc = None
    per_class_roc = None

    if y_prob is not None:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        try:
            roc_auc = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            roc_auc = None

        per_class_roc = {}
        for i in range(num_classes):
            fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_prob[:, i])
            per_class_roc[i] = (fpr, tpr, thresholds)

    return ClassificationResults(
        accuracy=accuracy,
        per_class_report=report,
        confusion_matrix=cm,
        roc_auc=roc_auc,
        per_class_roc=per_class_roc,
    )
