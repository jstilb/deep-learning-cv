"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import ClassificationResults, compute_metrics


class TestComputeMetrics:
    """Tests for the metrics computation pipeline."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        results = compute_metrics(y_true, y_pred, num_classes=5)
        assert results.accuracy == 1.0

    def test_completely_wrong(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        results = compute_metrics(y_true, y_pred, num_classes=2)
        assert results.accuracy == 0.0

    def test_confusion_matrix_shape(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        results = compute_metrics(y_true, y_pred, num_classes=3)
        assert results.confusion_matrix.shape == (3, 3)

    def test_confusion_matrix_diagonal_for_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        results = compute_metrics(y_true, y_pred, num_classes=3)
        np.testing.assert_array_equal(
            np.diag(results.confusion_matrix), [2, 2, 2]
        )

    def test_roc_auc_with_probabilities(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ])
        results = compute_metrics(y_true, y_pred, y_prob, num_classes=2)
        assert results.roc_auc is not None
        assert results.roc_auc > 0.9

    def test_roc_auc_none_without_probabilities(self) -> None:
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        results = compute_metrics(y_true, y_pred, num_classes=3)
        assert results.roc_auc is None
        assert results.per_class_roc is None

    def test_per_class_report_has_expected_keys(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        results = compute_metrics(
            y_true, y_pred, class_names=["cat", "dog"], num_classes=2
        )
        assert "cat" in results.per_class_report
        assert "dog" in results.per_class_report
        assert "precision" in results.per_class_report["cat"]

    def test_returns_classification_results_type(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        results = compute_metrics(y_true, y_pred, num_classes=2)
        assert isinstance(results, ClassificationResults)

    def test_torch_tensor_input(self) -> None:
        import torch

        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1, 2])
        results = compute_metrics(y_true, y_pred, num_classes=3)
        assert results.accuracy == 1.0
